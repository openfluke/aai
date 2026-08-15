package main

import (
	"strings"
	"testing"

	"github.com/openfluke/welvet/layers/parallel"
)

func TestPickSurvivorsFiltersDeadCluster(t *testing.T) {
	mk := func(kind CellKind, n int, pix float64) trainedNet {
		return trainedNet{
			job:    job{kind: kind, nHemi: n},
			stack:  &parallel.Stack{},
			result: &ArchResult{Train: SplitScore{MeanPixel: pix}},
		}
	}
	nets := []trainedNet{
		mk(KindDense, 3, 5.0),
		mk(KindResidual, 3, 10.4),
		mk(KindCNN1, 3, 5.1),
		mk(KindMHA, 3, 7.2),
		mk(KindDense, 2, 12.0), // other width
		mk(KindLSTM, 3, 4.9),
	}
	got := pickSurvivors(nets, 3, 6.0, 8)
	if len(got) != 2 {
		t.Fatalf("kept %d want 2 (residual+mha)", len(got))
	}
	if got[0].job.kind != KindResidual || got[1].job.kind != KindMHA {
		t.Fatalf("order %s then %s", got[0].job.kind, got[1].job.kind)
	}

	dead := []trainedNet{
		mk(KindDense, 1, 5.0),
		mk(KindCNN1, 1, 5.1),
		mk(KindMHA, 1, 4.8),
	}
	fb := pickSurvivors(dead, 1, 6.0, 8)
	if len(fb) != 2 {
		t.Fatalf("fallback kept %d want 2", len(fb))
	}
}

func TestHierSandwichTrain(t *testing.T) {
	const hidden = 32
	dim := 32
	mode := parallel.ModeStepTweenChain
	x := inputTensor(make([]float32, dim))
	y := inputTensor(make([]float32, dim))
	for i := range x.Data {
		x.Data[i] = 0.1
		y.Data[i] = 0.2
	}

	var surv []trainedNet
	for i, kind := range []CellKind{KindDense, KindResidual, KindCNN1} {
		st, err := buildNativeCameral(kind, dim, hidden, dim, 2, mode)
		if err != nil {
			t.Fatalf("build %s: %v", kind, err)
		}
		loss, err := parallel.TrainStackMSE(st, x, y, mode, 0.01)
		if err != nil {
			t.Fatalf("train %s: %v", kind, err)
		}
		if loss != loss {
			t.Fatalf("NaN %s", kind)
		}
		surv = append(surv, trainedNet{
			job:   job{kind: kind, nHemi: 2},
			stack: st,
			result: &ArchResult{
				Train: SplitScore{MeanPixel: 6.0 + float64(i)},
			},
		})
	}

	stack, kept, note, err := buildHierSandwich(surv, dim, hidden, dim, mode)
	if err != nil {
		t.Fatalf("hier build: %v", err)
	}
	if stack == nil || kept == "" {
		t.Fatalf("empty hier kept=%q note=%q", kept, note)
	}
	if !strings.Contains(kept, "dense-like") || !strings.Contains(kept, "conv") {
		t.Fatalf("want two families, kept=%s note=%s", kept, note)
	}
	_, post, err := parallel.ForwardStack(stack, x)
	if err != nil {
		t.Fatalf("hier forward: %v", err)
	}
	if post == nil || post.Len() != dim {
		t.Fatalf("hier out len %v", post)
	}
	loss, err := parallel.TrainStackMSE(stack, x, y, mode, 0.01)
	if err != nil {
		t.Fatalf("hier train: %v", err)
	}
	if loss != loss {
		t.Fatal("hier NaN loss")
	}
}
