package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

func allKinds() []CellKind {
	return append([]CellKind{KindDense}, allKindsExceptDense()...)
}

// allKindsExceptDense is every cameral kind except Dense / embedding.
func allKindsExceptDense() []CellKind {
	return []CellKind{
		KindCNN1, KindCNN2, KindCNN3,
		KindConvT1, KindConvT2, KindConvT3,
		KindMHA, KindLSTM, KindRNN,
		KindMamba, KindGDN, KindSwiGLU,
		KindResidual, KindSequential,
		KindSoftmax, KindLayerNorm, KindRMSNorm,
		KindKMeans, KindMetacognition,
	}
}

func parseLayerList(layerFlag, layersFlag string, args []string) ([]CellKind, error) {
	var raw []string
	for _, p := range splitLayerTokens(layersFlag) {
		raw = append(raw, p)
	}
	for _, a := range args {
		raw = append(raw, splitLayerTokens(a)...)
	}
	if len(raw) == 0 {
		k, err := parseCellKind(layerFlag)
		if err != nil {
			return nil, err
		}
		return []CellKind{k}, nil
	}
	if len(raw) == 1 {
		s := strings.ToLower(raw[0])
		if s == "all" {
			return allKinds(), nil
		}
		if s == "except-dense" || s == "nodense" {
			return allKindsExceptDense(), nil
		}
	}
	out := make([]CellKind, 0, len(raw))
	seen := map[CellKind]bool{}
	for _, s := range raw {
		k, err := parseCellKind(s)
		if err != nil {
			return nil, err
		}
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, k)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no layers to run")
	}
	return out, nil
}

func splitLayerTokens(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var out []string
	for _, p := range strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == ' ' || r == ';' || r == '|'
	}) {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func kindsCSV(kinds []CellKind) string {
	parts := make([]string, len(kinds))
	for i, k := range kinds {
		parts[i] = string(k)
	}
	return strings.Join(parts, ",")
}

func parseDTypeList(s string) ([]core.DType, error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "all") {
		out := make([]core.DType, len(core.AllDTypes))
		copy(out, core.AllDTypes)
		return out, nil
	}
	out := make([]core.DType, 0)
	seen := map[core.DType]bool{}
	for _, p := range splitLayerTokens(s) {
		dt, err := parseDTypeToken(p)
		if err != nil {
			return nil, err
		}
		if seen[dt] {
			continue
		}
		seen[dt] = true
		out = append(out, dt)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no dtypes")
	}
	return out, nil
}

func parseDTypeToken(p string) (core.DType, error) {
	p = strings.ToLower(strings.TrimSpace(p))
	if p == "" {
		return 0, fmt.Errorf("empty dtype")
	}
	switch p {
	case "f32", "float32", "fp32":
		return core.DTypeFloat32, nil
	case "f64", "float64", "fp64", "double":
		return core.DTypeFloat64, nil
	case "f16", "float16", "fp16":
		return core.DTypeFloat16, nil
	case "bf16", "bfloat16":
		return core.DTypeBFloat16, nil
	}
	for _, dt := range core.AllDTypes {
		if strings.ToLower(dt.String()) == p {
			return dt, nil
		}
	}
	if n, err := strconv.Atoi(p); err == nil && n >= 0 && n <= 33 {
		return core.DType(n), nil
	}
	return 0, fmt.Errorf("unknown dtype %q", p)
}

func dtypesCSV(dts []core.DType) string {
	parts := make([]string, len(dts))
	for i, dt := range dts {
		parts[i] = dt.String()
	}
	return strings.Join(parts, ",")
}

func jobName(kind CellKind, nHemi int) string {
	return string(kind) + "/" + camName(nHemi)
}

// newHemisphereOp builds one H→H brain. Spatial/seq kinds are wrapped in
// View → Op → View so Parallel still sees a [1, hidden] vector from the Dense stem.
func newHemisphereOp(kind CellKind, hidden int) (any, error) {
	if hidden <= 0 {
		return nil, fmt.Errorf("hidden must be > 0")
	}
	switch kind {
	case KindDense:
		d, err := dense.NewConfigured[float32](hidden, hidden, core.ActivationLeakyReLU,
			core.DTypeFloat32, quant.FormatNone, xavier(hidden, hidden))
		if err != nil {
			return nil, err
		}
		return d, nil
	case KindCNN1:
		op, err := cnn1.New(cnn1.Config{
			InChannels: 1, Filters: 1, SeqLen: hidden, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, 1, hidden}, op)
	case KindCNN2:
		c, h, w := factorCHW(hidden)
		op, err := cnn2.New(cnn2.Config{
			InChannels: c, Filters: c, Height: h, Width: w, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, c, h, w}, op)
	case KindCNN3:
		c, d, h, w := factorCDHW(hidden)
		op, err := cnn3.New(cnn3.Config{
			InChannels: c, Filters: c, Depth: d, Height: h, Width: w, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, c, d, h, w}, op)
	case KindConvT1:
		op, err := convt1.New(convt1.Config{
			InChannels: 1, Filters: 1, SeqLen: hidden, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, 1, hidden}, op)
	case KindConvT2:
		c, h, w := factorCHW(hidden)
		op, err := convt2.New(convt2.Config{
			InChannels: c, Filters: c, Height: h, Width: w, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, c, h, w}, op)
	case KindConvT3:
		c, d, h, w := factorCDHW(hidden)
		op, err := convt3.New(convt3.Config{
			InChannels: c, Filters: c, Depth: d, Height: h, Width: w, Kernel: 3, Stride: 1, Padding: 1,
			Activation: core.ActivationReLU,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, c, d, h, w}, op)
	case KindMHA:
		seq, dim := factor2(hidden)
		heads := mhaHeads(dim)
		op, err := mha.New(mha.Config{
			DModel: dim, NumHeads: heads, MaxSeqLen: seq,
			Mask: mha.MaskBidirectional, Pos: mha.PosNone, Mode: mha.ModeSelf,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, seq, dim}, op)
	case KindLSTM:
		seq, dim := factor2(hidden)
		op, err := lstm.New(lstm.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, seq, dim}, op)
	case KindRNN:
		seq, dim := factor2(hidden)
		op, err := rnn.New(rnn.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, seq, dim}, op)
	case KindMamba:
		seq, dim := factor2(hidden)
		op, err := mamba.New(mamba.Config{DModel: dim, DState: minPos(dim, 16), SeqLen: seq, Expand: 2})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return wrapRank(hidden, []int{1, seq, dim}, op)
	case KindGDN:
		seq, dim := factor2(hidden)
		heads := mhaHeads(dim)
		hd := dim / heads
		op, err := gdn.NewConfigured(gdn.Config{
			HiddenSize: dim, NumKeyHeads: heads, NumValueHeads: heads,
			KeyHeadDim: hd, ValueHeadDim: hd, ConvKernel: 4,
		},
			xavier(heads*hd*2+heads*hd, dim),
			xavier(heads*hd, dim),
			xavier(heads, dim),
			xavier(heads, dim),
			xavier(dim, heads*hd),
			xavier((heads*hd*2+heads*hd), 4),
			nil, nil, nil)
		if err != nil {
			return nil, err
		}
		return wrapRank(hidden, []int{1, seq, dim}, op)
	case KindSwiGLU:
		op, err := swiglu.New(swiglu.DefaultFFN(hidden))
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return op, nil
	case KindResidual:
		op, err := residual.New(residual.Config{Dim: hidden, Depth: 1})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return op, nil
	case KindSequential:
		op, err := sequential.New(sequential.Config{Dim: hidden, Depth: 2})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return op, nil
	case KindSoftmax:
		return softmax.New(softmax.Config{Dim: hidden})
	case KindLayerNorm:
		return layernorm.New(layernorm.Config{Dim: hidden})
	case KindRMSNorm:
		return rmsnorm.New(rmsnorm.Config{Dim: hidden})
	case KindKMeans:
		op, err := kmeans.New(kmeans.Config{
			NumClusters: hidden, FeatureDim: hidden,
			OutputMode: kmeans.OutputFeatures,
		})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return op, nil
	case KindMetacognition:
		op, err := metacognition.New(metacognition.Config{Dim: hidden})
		if err != nil {
			return nil, err
		}
		xavierStores(op)
		return op, nil
	case KindEmbedding:
		return nil, fmt.Errorf("embedding needs token ids — skip for ARC vectors")
	default:
		return nil, fmt.Errorf("layer %q not wired", kind)
	}
}

func wrapRank(hidden int, inShape []int, op any) (any, error) {
	if op == nil {
		return nil, fmt.Errorf("nil hemisphere op")
	}
	if len(inShape) == 2 && inShape[0] == 1 && inShape[1] == hidden {
		return op, nil
	}
	inV, err := parallel.NewView(inShape...)
	if err != nil {
		return nil, err
	}
	outV, err := parallel.NewView(1, hidden)
	if err != nil {
		return nil, err
	}
	return parallel.NewStack(inV, op, outV)
}

func xavierStores(op any) {
	for _, st := range dna.CollectStores(op) {
		if st == nil || st.Rows <= 1 || st.Cols <= 0 {
			continue
		}
		_ = st.SetFromF32(xavier(st.Rows, st.Cols))
	}
}

func mhaHeads(d int) int {
	for _, h := range []int{4, 2, 1} {
		if h <= d && d%h == 0 && d/h >= 2 {
			return h
		}
	}
	return 1
}

func factor2(n int) (h, w int) {
	if n <= 0 {
		return 1, 1
	}
	h = int(math.Sqrt(float64(n)))
	for h > 1 && n%h != 0 {
		h--
	}
	if h < 1 {
		h = 1
	}
	return h, n / h
}

func factorCHW(n int) (c, h, w int) {
	bestC, bestH, bestW := 1, 1, n
	best := n + 1
	for cc := 1; cc <= n; cc++ {
		if n%cc != 0 {
			continue
		}
		hh, ww := factor2(n / cc)
		if hh < 2 || ww < 2 {
			continue
		}
		diff := hh - ww
		if diff < 0 {
			diff = -diff
		}
		score := diff*20 + absInt(cc-hh)
		if score < best {
			best, bestC, bestH, bestW = score, cc, hh, ww
		}
	}
	return bestC, bestH, bestW
}

func factorCDHW(n int) (c, d, h, w int) {
	best := n + 1
	bestC, bestD, bestH, bestW := 1, 1, 1, n
	for cc := 1; cc <= n; cc++ {
		if n%cc != 0 {
			continue
		}
		rest := n / cc
		dd := int(math.Cbrt(float64(rest))) + 1
		for ; dd >= 1; dd-- {
			if rest%dd != 0 {
				continue
			}
			hh, ww := factor2(rest / dd)
			if hh < 2 || ww < 2 || dd < 2 {
				continue
			}
			span := max3(dd, hh, ww) - min3(dd, hh, ww)
			score := span*20 + absInt(cc-hh)
			if score < best {
				best, bestC, bestD, bestH, bestW = score, cc, dd, hh, ww
			}
		}
	}
	return bestC, bestD, bestH, bestW
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minPos(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func min3(a, b, c int) int {
	if a <= b && a <= c {
		return a
	}
	if b <= c {
		return b
	}
	return c
}

func max3(a, b, c int) int {
	if a >= b && a >= c {
		return a
	}
	if b >= c {
		return b
	}
	return c
}
