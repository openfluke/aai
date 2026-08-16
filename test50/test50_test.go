package main

import (
	"testing"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

func TestParseModesAll(t *testing.T) {
	ms, err := parseModes("all")
	if err != nil {
		t.Fatal(err)
	}
	want := parallel.AllNamedTrainModes()
	if len(ms) != len(want) {
		t.Fatalf("all modes %d want %d", len(ms), len(want))
	}
	foundBP, foundFast, foundSparse, foundMesh := false, false, false, false
	for _, m := range ms {
		if m == parallel.ModeInherit {
			t.Fatal("all leaked Inherit")
		}
		switch m {
		case parallel.ModeStepBP:
			foundBP = true
		case parallel.ModeTweenSplitFastProxy:
			foundFast = true
		case parallel.ModeTweenSplitSparse:
			foundSparse = true
		case parallel.ModeMeshBP, parallel.ModeMeshTweenSplitFastProxy:
			foundMesh = true
		}
	}
	if !foundBP || !foundFast || !foundSparse || !foundMesh {
		t.Fatal("all should include StepBP, FastProxy, Sparse, Mesh*")
	}
}

func TestParseModesMesh(t *testing.T) {
	ms, err := parseModes("meshbp,meshfastproxy,meshsparse")
	if err != nil {
		t.Fatal(err)
	}
	want := []parallel.TrainMode{
		parallel.ModeMeshBP,
		parallel.ModeMeshTweenSplitFastProxy,
		parallel.ModeMeshTweenSplitSparse,
	}
	if len(ms) != len(want) {
		t.Fatalf("got %v", ms)
	}
	for i := range want {
		if ms[i] != want[i] {
			t.Fatalf("got %v want %v", ms, want)
		}
	}
}

func TestParseGrids(t *testing.T) {
	got, err := parseGrids("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("%v", got)
	}
	got, err = parseGrids("1x1x1,3")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("%v", got)
	}
}

func TestParseModesOptIn(t *testing.T) {
	ms, err := parseModes("stepbp,tweensplit,fastproxy,sparse")
	if err != nil {
		t.Fatal(err)
	}
	want := []parallel.TrainMode{
		parallel.ModeStepBP, parallel.ModeTweenSplit,
		parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitSparse,
	}
	if len(ms) != len(want) {
		t.Fatalf("got %v", ms)
	}
	for i := range want {
		if ms[i] != want[i] {
			t.Fatalf("got %v want %v", ms, want)
		}
	}
}

func TestXORMeshSmoke(t *testing.T) {
	task := makeXOR()
	for _, gn := range []int{1, 3} {
		for _, mode := range []parallel.TrainMode{
			parallel.ModeMeshBP,
			parallel.ModeMeshTween,
			parallel.ModeMeshTweenSplitFastProxy,
		} {
			j := job{task: task, kind: KindDense, nHemi: 1, mode: mode, dt: core.DTypeFloat32, gridN: gn}
			r := runLucyJob(j, 32, 80*time.Millisecond, 40*time.Millisecond, 0.05, 1, 2)
			if r.Err != "" {
				t.Fatalf("%s %s: %s", gridName(gn), mode, r.Err)
			}
			if r.Steps < 1 {
				t.Fatalf("%s %s no steps", gridName(gn), mode)
			}
			if r.Grid != gridName(gn) {
				t.Fatalf("grid %s want %s", r.Grid, gridName(gn))
			}
		}
	}
}

func TestXORStepBPAndFastProxySmoke(t *testing.T) {
	task := makeXOR()
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepBP,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitSparse,
	} {
		j := job{task: task, kind: KindDense, nHemi: 1, mode: mode, dt: core.DTypeFloat32}
		r := runLucyJob(j, 32, 80*time.Millisecond, 40*time.Millisecond, 0.05, 1, 2)
		if r.Err != "" {
			t.Fatalf("%s: %s", mode, r.Err)
		}
		if r.Soft != r.Soft || r.Score != r.Score {
			t.Fatalf("%s NaN", mode)
		}
		if r.Steps < 1 {
			t.Fatalf("%s no steps", mode)
		}
		if r.DType != "float32" {
			t.Fatalf("dtype %s", r.DType)
		}
	}
}

func TestSineSwitchSmoke(t *testing.T) {
	task := makeSineAdapt()
	if len(task.pools) != 4 {
		t.Fatalf("sine pools %d", len(task.pools))
	}
	j := job{task: task, kind: KindDense, nHemi: 2, mode: parallel.ModeStepBP, dt: core.DTypeFloat32}
	r := runLucyJob(j, 32, 80*time.Millisecond, 20*time.Millisecond, 0.05, 1, 2)
	if r.Err != "" {
		t.Fatal(r.Err)
	}
}

func TestCollectWinners(t *testing.T) {
	rows := []row{
		{Task: "sine", DType: "float32", Layer: "dense", Arch: "Dense", Mode: "StepBP", Acc: 40, Score: 100},
		{Task: "sine", DType: "float32", Layer: "dense", Arch: "Dense", Mode: "TweenSplitFastProxy", Acc: 80, Score: 90},
		{Task: "sine", DType: "float32", Layer: "dense", Arch: "Dense", Mode: "TweenSplitSparse", Acc: 20, Score: 200},
	}
	w := collectWinners(rows)
	if len(w) != 2 {
		t.Fatalf("winners %d want 2 (Acc + Score)", len(w))
	}
	if w[0].Axis != "hard Acc" || w[0].Mode != "TweenSplitFastProxy" {
		t.Fatalf("Acc winner %+v", w[0])
	}
	if w[1].Axis != "Lucy Score" || w[1].Mode != "TweenSplitSparse" {
		t.Fatalf("Score winner %+v", w[1])
	}
}
