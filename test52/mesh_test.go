package main

import (
	"math/rand"
	"testing"
	"time"

	"github.com/openfluke/welvet/layers/parallel"
)

func TestPlaceIs3x3x3(t *testing.T) {
	st, err := buildMLP(rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	g, err := placeOn3x3x3(st)
	if err != nil {
		t.Fatal(err)
	}
	if g.Depth != 3 || g.Rows != 3 || g.Cols != 3 {
		t.Fatalf("got %d×%d×%d", g.Depth, g.Rows, g.Cols)
	}
	live := 0
	for i := range g.Cells {
		if !g.Cells[i].Layer.IsDisabled {
			live++
		}
	}
	if live != 1 {
		t.Fatalf("live cells=%d want 1 (origin)", live)
	}
}

func TestMeshVolumeIs3x3x3(t *testing.T) {
	g, dens, err := buildMeshVolume3(rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	if g.Depth != 3 || g.Rows != 3 || g.Cols != 3 {
		t.Fatalf("got %d×%d×%d", g.Depth, g.Rows, g.Cols)
	}
	if len(dens) != 27 {
		t.Fatalf("dens=%d want 27", len(dens))
	}
	live := 0
	for i := range g.Cells {
		if !g.Cells[i].Layer.IsDisabled {
			live++
		}
	}
	if live != 27 {
		t.Fatalf("live cells=%d want 27 (full 3³)", live)
	}
}

func TestNormalBPLearnsOn3x3x3(t *testing.T) {
	r := smokeMode(parallel.ModeNormalBP, 7, 800*time.Millisecond, 0.05, 64)
	if !r.Pass {
		t.Fatalf("NormalBP FAIL: err=%q loss %.4f→%.4f Acc %.0f→%.0f Δw=%.3g",
			r.Err, r.Loss0, r.Loss1, r.Acc0, r.Acc1, r.WDelta)
	}
}

func TestMeshBPLearnsOn3x3x3(t *testing.T) {
	r := smokeMode(parallel.ModeMeshBP, 11, 800*time.Millisecond, 0.05, 64)
	if r.Err != "" {
		t.Fatalf("MeshBP err: %s", r.Err)
	}
	if !r.Pass {
		t.Fatalf("MeshBP FAIL loss %.4f→%.4f Acc %.0f→%.0f Δw=%.3g",
			r.Loss0, r.Loss1, r.Acc0, r.Acc1, r.WDelta)
	}
}

func TestMeshTweenLearnsOn3x3x3(t *testing.T) {
	r := smokeMode(parallel.ModeMeshTween, 13, 1200*time.Millisecond, 0.05, 64)
	if r.Err != "" {
		t.Fatalf("MeshTween err: %s", r.Err)
	}
	if !r.Pass {
		t.Fatalf("MeshTween FAIL loss %.4f→%.4f Acc %.0f→%.0f Δw=%.3g layout=%s",
			r.Loss0, r.Loss1, r.Acc0, r.Acc1, r.WDelta, r.Layout)
	}
}
