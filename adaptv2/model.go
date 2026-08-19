package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

const (
	inDim  = 8
	outDim = 4
)

// loom test [2] widths: 8→32→64→64→64→32→4
var layerWidths = []int{inDim, 32, 64, 64, 64, 32, outDim}

func buildChaseNet(rng *rand.Rand, be core.Backend) (*parallel.Stack, error) {
	ops := make([]any, 0, len(layerWidths)-1)
	for i := 0; i < len(layerWidths)-1; i++ {
		in, out := layerWidths[i], layerWidths[i+1]
		act := core.ActivationTanh
		if i == len(layerWidths)-2 {
			act = core.ActivationLinear
		}
		w := randScale(in*out, float32(1/math.Sqrt(float64(in))), rng)
		d, err := dense.NewConfigured(in, out, act, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, fmt.Errorf("dense %d: %w", i, err)
		}
		ops = append(ops, d)
	}
	st, err := parallel.NewStack(ops...)
	if err != nil {
		return nil, err
	}
	st.Exec.Backend = be
	st.Exec.MultiCore = true
	st.SyncChildExec()
	return st, nil
}

func randScale(n int, scale float32, rng *rand.Rand) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * scale
	}
	return w
}

type teacher struct {
	w []float32 // [outDim, inDim]
}

func newTeacher(rng *rand.Rand) teacher {
	return teacher{w: randScale(outDim*inDim, 1, rng)}
}

func (t teacher) classOf(x []float32) int {
	best, bv := 0, float32(-1e9)
	for c := 0; c < outDim; c++ {
		var s float32
		for i := 0; i < inDim; i++ {
			s += t.w[c*inDim+i] * x[i]
		}
		if s > bv {
			bv, best = s, c
		}
	}
	return best
}

func sampleX(rng *rand.Rand) *core.Tensor[float32] {
	x := core.NewTensor[float32](1, inDim)
	for i := range x.Data {
		x.Data[i] = rng.Float32()*2 - 1
	}
	return x
}

func oneHot(class int) *core.Tensor[float32] {
	y := core.NewTensor[float32](1, outDim)
	if class < 0 {
		class = 0
	}
	y.Data[class%outDim] = 1
	return y
}

func labelFor(t teacher, x *core.Tensor[float32], avoid bool) *core.Tensor[float32] {
	c := t.classOf(x.Data)
	if avoid {
		c = (c + 2) % outDim
	}
	return oneHot(c)
}

func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

func softmaxPTrue(logits []float32, lab int) float32 {
	if lab < 0 || lab >= len(logits) {
		return 0
	}
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	exps := make([]float32, len(logits))
	for i, v := range logits {
		exps[i] = float32(math.Exp(float64(v - max)))
		sum += exps[i]
	}
	if sum <= 0 {
		return 1 / float32(len(logits))
	}
	return exps[lab] / sum
}

func stackWeightBytes(s *parallel.Stack) int64 {
	var n int64
	for _, st := range dna.CollectStores(s) {
		if st == nil {
			continue
		}
		elems := int64(st.Rows * st.Cols)
		if elems <= 0 {
			f, err := st.FlattenF32()
			if err != nil || len(f) == 0 {
				continue
			}
			elems = int64(len(f))
		}
		bits := int64(st.DType.Bits())
		if bits <= 0 {
			bits = 32
		}
		n += (elems*bits + 7) / 8
	}
	return n
}
