package main

import (
	"testing"

	"github.com/openfluke/welvet/layers/parallel"
)

func TestParseJobModesAllIncludesSplitAlt(t *testing.T) {
	seq, mesh, err := parseJobModes("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(seq) != 10 {
		t.Fatalf("seq %d want 10", len(seq))
	}
	if len(mesh) != 3 {
		t.Fatalf("mesh %d want 3", len(mesh))
	}
	found := map[TrainingMode]bool{}
	for _, m := range seq {
		found[m] = true
	}
	for _, m := range []TrainingMode{ModeTweenSplit, ModeStepTweenSplit, ModeTweenAlt, ModeStepTweenAlt} {
		if !found[m] {
			t.Fatalf("missing %s", modeNames[m])
		}
	}
}

func TestParseJobModesSplitAltOnly(t *testing.T) {
	seq, mesh, err := parseJobModes("tweensplit,steptweensplit,tweenalt,steptweenalt")
	if err != nil {
		t.Fatal(err)
	}
	if len(mesh) != 0 {
		t.Fatalf("mesh %v", mesh)
	}
	want := []TrainingMode{ModeTweenSplit, ModeStepTweenSplit, ModeTweenAlt, ModeStepTweenAlt}
	if len(seq) != len(want) {
		t.Fatalf("got %v", seq)
	}
	for i := range want {
		if seq[i] != want[i] {
			t.Fatalf("got %v want %v", seq, want)
		}
	}
}

func TestBuildJobsUniformCount(t *testing.T) {
	seq, mesh, err := parseJobModes("all")
	if err != nil {
		t.Fatal(err)
	}
	jobs := buildJobs("off", seq, mesh)
	if len(jobs) != 43 {
		t.Fatalf("uniform jobs %d want 43", len(jobs))
	}
	stackN := 0
	for _, j := range jobs {
		if isStackCreditMode(j.mode) {
			stackN++
		}
	}
	if stackN != 16 {
		t.Fatalf("stack credit jobs %d want 16", stackN)
	}
}

func TestToParallelMode(t *testing.T) {
	pm, ok := toParallelMode(ModeTweenAlt)
	if !ok || pm != parallel.ModeTweenAlt {
		t.Fatalf("got %v %v", pm, ok)
	}
	if _, ok := toParallelMode(ModeStepBP); ok {
		t.Fatal("StepBP is Grid, not Stack credit")
	}
}
