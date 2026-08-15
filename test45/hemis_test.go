package main

import (
	"testing"

	"github.com/openfluke/welvet/layers/parallel"
)

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
	nodense, err := parseLayerList("dense", "except-dense", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range nodense {
		if k == KindDense {
			t.Fatal("except-dense should skip dense")
		}
	}
	got, err := parseLayerList("dense", "", []string{"cnn", "cnn2", "cnn3", "mha", "lstm"})
	if err != nil {
		t.Fatal(err)
	}
	want := []CellKind{KindCNN1, KindCNN2, KindCNN3, KindMHA, KindLSTM}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestHemisphereSandwichTrain(t *testing.T) {
	const hidden = 64
	dim := 64
	x := inputTensor(make([]float32, dim))
	y := inputTensor(make([]float32, dim))
	for i := range x.Data {
		x.Data[i] = 0.1
		y.Data[i] = 0.2
	}
	for _, kind := range allKinds() {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			stack, err := buildNativeCameral(kind, dim, hidden, dim, 3, parallel.ModeStepTweenChain)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			_, post, err := parallel.ForwardStack(stack, x)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			if post == nil || post.Len() != dim {
				t.Fatalf("out len %v", post)
			}
			loss, err := parallel.TrainStackMSE(stack, x, y, parallel.ModeStepTweenChain, 0.01)
			if err != nil {
				t.Fatalf("train: %v", err)
			}
			if loss != loss {
				t.Fatal("NaN loss")
			}
		})
	}
}
