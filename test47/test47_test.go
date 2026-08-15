package main

import (
	"testing"
	"time"

	"github.com/openfluke/welvet/layers/parallel"
)

func TestParseModes(t *testing.T) {
	ms, err := parseModes("steptween,tween,tweensplit,steptweensplit,tweenalt,steptweenalt,alt")
	if err != nil {
		t.Fatal(err)
	}
	want := []parallel.TrainMode{
		parallel.ModeStepTween, parallel.ModeTween,
		parallel.ModeTweenSplit, parallel.ModeStepTweenSplit,
		parallel.ModeTweenAlt, parallel.ModeStepTweenAlt,
	}
	if len(ms) != len(want) {
		t.Fatalf("got %v", ms)
	}
	for i := range want {
		if ms[i] != want[i] {
			t.Fatalf("got %v want %v", ms, want)
		}
	}
	if ms[0].Family() != ms[1].Family() {
		t.Fatal("Tween and StepTween should share a family")
	}
	if ms[2].Family() != ms[3].Family() {
		t.Fatal("TweenSplit and StepTweenSplit should share a family")
	}
	if ms[4].Family() != ms[5].Family() {
		t.Fatal("TweenAlt and StepTweenAlt should share a family")
	}
	if ms[0].Family() == ms[2].Family() {
		t.Fatal("tween family must differ from split family")
	}
	if ms[2].Family() == ms[4].Family() {
		t.Fatal("split family must differ from alt family")
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

func TestXORTweenVsSplitSmoke(t *testing.T) {
	task := makeXOR()
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepTween, parallel.ModeTween,
		parallel.ModeTweenSplit, parallel.ModeStepTweenSplit,
		parallel.ModeTweenAlt, parallel.ModeStepTweenAlt,
	} {
		j := job{task: task, kind: KindDense, nHemi: 1, mode: mode}
		r := runJob(j, 32, 80*time.Millisecond, 0.05, 2)
		if r.Err != "" {
			t.Fatalf("%s: %s", mode, r.Err)
		}
		if r.Soft != r.Soft {
			t.Fatalf("%s NaN soft", mode)
		}
	}
}
