package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

// CellKind selects the Op family used for stem / hemispheres / head.
// Dense is wired now. Other welvet kinds are named so later sweeps
// can swap HemispheresFrom branches without changing the ARC loop.
type CellKind string

const (
	KindDense         CellKind = "dense"
	KindCNN1          CellKind = "cnn1"
	KindCNN2          CellKind = "cnn2"
	KindCNN3          CellKind = "cnn3"
	KindMHA           CellKind = "mha"
	KindRNN           CellKind = "rnn"
	KindLSTM          CellKind = "lstm"
	KindMamba         CellKind = "mamba"
	KindGDN           CellKind = "gdn"
	KindSwiGLU        CellKind = "swiglu"
	KindResidual      CellKind = "residual"
	KindSequential    CellKind = "sequential"
	KindSoftmax       CellKind = "softmax"
	KindLayerNorm     CellKind = "layernorm"
	KindRMSNorm       CellKind = "rmsnorm"
	KindEmbedding     CellKind = "embedding"
	KindKMeans        CellKind = "kmeans"
	KindMetacognition CellKind = "metacognition"
)

func parseCellKind(s string) (CellKind, error) {
	k := CellKind(strings.ToLower(strings.TrimSpace(s)))
	switch k {
	case KindDense, KindCNN1, KindCNN2, KindCNN3, KindMHA, KindRNN, KindLSTM,
		KindMamba, KindGDN, KindSwiGLU, KindResidual, KindSequential, KindSoftmax,
		KindLayerNorm, KindRMSNorm, KindEmbedding, KindKMeans, KindMetacognition:
		return k, nil
	default:
		return "", fmt.Errorf("unknown layer %q (dense now; others reserved)", s)
	}
}

func camName(nHemi int) string {
	switch nHemi {
	case 0, 1:
		return "Dense"
	case 2:
		return "Bicameral"
	case 3:
		return "Tricameral"
	case 4:
		return "Quadcameral"
	default:
		return fmt.Sprintf("%d-cameral", nHemi)
	}
}

// buildNativeCameral is Dense(in→hidden) → native Hemispheres(n, add) → Dense(hidden→out).
// nHemi≤1 is a Dense sandwich (no Parallel) so the baseline shares stem/head geometry.
// Future kinds go through HemispheresFrom once branch builders exist.
func buildNativeCameral(kind CellKind, in, hidden, out, nHemi int, mode parallel.TrainMode) (*parallel.Stack, error) {
	if kind != KindDense {
		return nil, fmt.Errorf("test44: layer %q not wired yet — dense only for now (HemispheresFrom is ready)", kind)
	}
	if in <= 0 || hidden <= 0 || out <= 0 {
		return nil, fmt.Errorf("test44: need positive in/hidden/out")
	}
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU,
		core.DTypeFloat32, quant.FormatNone, xavier(hidden, in))
	if err != nil {
		return nil, fmt.Errorf("stem: %w", err)
	}

	var mid any
	if nHemi <= 1 {
		mid, err = dense.NewConfigured[float32](hidden, hidden, core.ActivationLeakyReLU,
			core.DTypeFloat32, quant.FormatNone, xavier(hidden, hidden))
		if err != nil {
			return nil, fmt.Errorf("dense mid: %w", err)
		}
	} else {
		hemi, herr := parallel.Hemispheres(hidden, hidden, nHemi, parallel.CombineAdd,
			core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
		if herr != nil {
			return nil, fmt.Errorf("hemispheres n=%d: %w", nHemi, herr)
		}
		for i := 0; i < nHemi; i++ {
			br, ok := hemi.DenseBranch(i)
			if !ok || br == nil || br.Weights == nil {
				return nil, fmt.Errorf("hemisphere %d not Dense", i)
			}
			if err := br.Weights.SetFromF32(xavier(hidden, hidden)); err != nil {
				return nil, fmt.Errorf("hemisphere %d init: %w", i, err)
			}
		}
		modes := make([]parallel.TrainMode, nHemi)
		for i := range modes {
			modes[i] = mode
		}
		hemi.SetBranchModes(modes...)
		mid = hemi
	}

	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationSigmoid,
		core.DTypeFloat32, quant.FormatNone, xavier(out, hidden))
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}

	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		return nil, err
	}
	s.Exec.Backend = core.BackendSIMD
	s.Exec.MultiCore = true
	s.Exec.TileSize = 32
	s.SyncChildExec()
	return s, nil
}

var rngMu sync.Mutex

func xavier(out, in int) []float32 {
	scale := float32(1)
	if in > 0 {
		scale = float32(1 / math.Sqrt(float64(in)))
	}
	w := make([]float32, out*in)
	rngMu.Lock()
	for i := range w {
		w[i] = (rand.Float32()*2 - 1) * scale
	}
	rngMu.Unlock()
	return w
}

func inputTensor(raw []float32) *core.Tensor[float32] {
	t := core.NewTensor[float32](1, len(raw))
	copy(t.Data, raw)
	return t
}
