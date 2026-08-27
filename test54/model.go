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
	"github.com/openfluke/welvet/systems/dna"
)

// CellKind selects the mid Op in Dense→mid→Dense.
type CellKind string

const (
	KindDense         CellKind = "dense"
	KindCNN1          CellKind = "cnn1"
	KindCNN2          CellKind = "cnn2"
	KindCNN3          CellKind = "cnn3"
	KindConvT1        CellKind = "convt1"
	KindConvT2        CellKind = "convt2"
	KindConvT3        CellKind = "convt3"
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
	case "cnn":
		return KindCNN1, nil
	case KindDense, KindCNN1, KindCNN2, KindCNN3, KindConvT1, KindConvT2, KindConvT3,
		KindMHA, KindRNN, KindLSTM, KindMamba, KindGDN, KindSwiGLU, KindResidual,
		KindSequential, KindSoftmax, KindLayerNorm, KindRMSNorm, KindEmbedding,
		KindKMeans, KindMetacognition:
		return k, nil
	default:
		return "", fmt.Errorf("unknown layer %q", s)
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
	default:
		return fmt.Sprintf("%d-cameral", nHemi)
	}
}

// leafMultiCore: false when many leaf-workers share the machine.
var leafMultiCore = true

// buildDeepSandwich: Dense(in→hidden) → mid×depth(kind) → Dense(hidden→out).
// depth is how many mid blocks of `kind` stack between stem and head.
// nCams>1 fans each mid Op into parallel branches (Welvet cameral merge).
// Mesh* modes place this stack on a fixed 1×1×1 origin grid at train time.
func buildSandwich(kind CellKind, in, hidden, out, depth, nCams int, dt core.DType) (*parallel.Stack, error) {
	if nCams < 1 {
		nCams = 1
	}
	if depth < 1 {
		depth = 1
	}
	if in <= 0 || hidden <= 0 || out <= 0 {
		return nil, fmt.Errorf("test54: need positive in/hidden/out")
	}
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU,
		core.DTypeFloat32, quant.FormatNone, xavier(hidden, in))
	if err != nil {
		return nil, fmt.Errorf("stem: %w", err)
	}
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationLinear,
		core.DTypeFloat32, quant.FormatNone, xavier(out, hidden))
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	ops := make([]any, 0, depth+2)
	ops = append(ops, stem)
	for i := 0; i < depth; i++ {
		mid, err := newMidBlock(kind, hidden, nCams)
		if err != nil {
			return nil, fmt.Errorf("mid[%d] %s: %w", i, kind, err)
		}
		ops = append(ops, mid)
	}
	ops = append(ops, head)
	s, err := parallel.Sandwich(ops...)
	if err != nil {
		return nil, err
	}
	if dt != core.DTypeFloat32 {
		if err := s.SetDType(dt); err != nil {
			return nil, fmt.Errorf("set dtype %s: %w", dt, err)
		}
	}
	s.Exec.Backend = core.BackendCPUTiled
	if kind == KindDense && dt == core.DTypeFloat32 {
		s.Exec.Backend = core.BackendSIMD
	}
	s.Exec.MultiCore = leafMultiCore
	s.Exec.TileSize = 32
	s.SyncChildExec()
	return s, nil
}

func newMidBlock(kind CellKind, hidden, nCams int) (any, error) {
	if nCams <= 1 {
		return newHemisphereOp(kind, hidden)
	}
	branches := make([]any, nCams)
	var err error
	for i := 0; i < nCams; i++ {
		branches[i], err = newHemisphereOp(kind, hidden)
		if err != nil {
			return nil, fmt.Errorf("hemi %d: %w", i, err)
		}
	}
	return parallel.HemispheresFrom(parallel.Config{
		Dim: hidden, OutFeat: hidden, Branches: nCams, Combine: parallel.CombineAdd,
	}, branches, nil)
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
