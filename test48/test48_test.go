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

func TestCameralCap(t *testing.T) {
	if camName(1) != "Dense" || camName(2) != "Bicameral" || camName(3) != "Tricameral" {
		t.Fatal(camName(1), camName(2), camName(3))
	}
}

func TestParseDTypesAll(t *testing.T) {
	dts, err := parseDTypeList("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(dts) != len(core.AllDTypes) {
		t.Fatalf("all dtypes %d want %d", len(dts), len(core.AllDTypes))
	}
	if dts[0] != core.DTypeFloat64 || dts[1] != core.DTypeFloat32 {
		t.Fatalf("order: %v %v", dts[0], dts[1])
	}
}

func TestParseDTypesOptIn(t *testing.T) {
	dts, err := parseDTypeList("float32,f16,int8")
	if err != nil {
		t.Fatal(err)
	}
	want := []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeInt8}
	if len(dts) != len(want) {
		t.Fatalf("got %v", dts)
	}
	for i := range want {
		if dts[i] != want[i] {
			t.Fatalf("got %v want %v", dts, want)
		}
	}
}

func TestParseDTypesUnknown(t *testing.T) {
	if _, err := parseDTypeList("notatype"); err == nil {
		t.Fatal("want error")
	}
}

func TestFloat16XORSmoke(t *testing.T) {
	task := makeXOR()
	j := job{task: task, kind: KindDense, nHemi: 1, mode: parallel.ModeStepBP, dt: core.DTypeFloat16}
	r := runLucyJob(j, 32, 80*time.Millisecond, 40*time.Millisecond, 0.05, 1, 2)
	if r.Err != "" {
		t.Fatal(r.Err)
	}
	if r.DType != "float16" {
		t.Fatalf("dtype %s", r.DType)
	}
	if r.Steps < 1 {
		t.Fatal("no steps")
	}
}

func TestWeightDTypeSmoke(t *testing.T) {
	task := makeXOR()
	for _, dt := range []core.DType{
		core.DTypeFloat64, core.DTypeFloat32, core.DTypeBFloat16,
		core.DTypeInt8, core.DTypeBinary, core.DTypeNF4,
	} {
		j := job{task: task, kind: KindDense, nHemi: 1, mode: parallel.ModeTweenSplitFastProxy, dt: dt}
		r := runLucyJob(j, 16, 40*time.Millisecond, 20*time.Millisecond, 0.05, 1, 2)
		if r.Err != "" {
			t.Errorf("%s: %s", dt, r.Err)
		}
		if r.DType != dt.String() {
			t.Errorf("%s: row dtype %s", dt, r.DType)
		}
	}
}
