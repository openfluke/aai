package main

import (
	"testing"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

func TestAdaptV2Smoke(t *testing.T) {
	r := runJob(parallel.ModeStepBP, core.BackendCPUTiled, 120*time.Millisecond, 40*time.Millisecond, 40*time.Millisecond, 3, 0.05, 1)
	if r.Err != "" {
		t.Fatal(r.Err)
	}
	if r.Outputs < 1 {
		t.Fatalf("no outputs: %+v", r)
	}
}

func TestParseModesLoom(t *testing.T) {
	ms, err := parseModeList("loom")
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 6 {
		t.Fatalf("loom modes %d", len(ms))
	}
}

func TestParseModesStep(t *testing.T) {
	ms, err := parseModeList("step")
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) < 5 {
		t.Fatalf("step modes %d", len(ms))
	}
	for _, m := range ms {
		if !m.IsLineStep() {
			t.Fatalf("%s is not a line step", m)
		}
	}
}
