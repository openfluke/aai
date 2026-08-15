package main

import (
	"testing"
	"time"

	"github.com/openfluke/welvet/layers/parallel"
)

func TestParseModesAll(t *testing.T) {
	ms, err := parseModes("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 16 {
		t.Fatalf("all modes %d want 16", len(ms))
	}
	foundBP, foundFast, foundSparse := false, false, false
	for _, m := range ms {
		switch m {
		case parallel.ModeStepBP:
			foundBP = true
		case parallel.ModeTweenSplitFastProxy:
			foundFast = true
		case parallel.ModeTweenSplitSparse:
			foundSparse = true
		}
	}
	if !foundBP || !foundFast || !foundSparse {
		t.Fatal("all should include StepBP, FastProxy, Sparse")
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

func TestParseLayersAllIncludesDense(t *testing.T) {
	ks, err := parseLayerList("dense", "all", nil)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, k := range ks {
		if k == KindDense {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("all should include dense")
	}
}

func TestXORStepBPAndFastProxySmoke(t *testing.T) {
	task := makeXOR()
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepBP,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitSparse,
	} {
		j := job{task: task, kind: KindDense, nHemi: 1, mode: mode}
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
	}
}

func TestSineSwitchSmoke(t *testing.T) {
	task := makeSineAdapt()
	if len(task.pools) != 4 {
		t.Fatalf("sine pools %d", len(task.pools))
	}
	j := job{task: task, kind: KindDense, nHemi: 2, mode: parallel.ModeStepBP}
	r := runLucyJob(j, 32, 80*time.Millisecond, 20*time.Millisecond, 0.05, 1, 2)
	if r.Err != "" {
		t.Fatal(r.Err)
	}
}

func TestCameralCap(t *testing.T) {
	if camName(1) != "Dense" || camName(2) != "Bicameral" || camName(3) != "Tricameral" {
		t.Fatal(camName(1), camName(2), camName(3))
	}
}
