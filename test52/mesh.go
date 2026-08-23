package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/dna"
)

const (
	inDim  = 2
	hidDim = 16
	outDim = 1
	gridN  = 3 // always 3×3×3 spatial
	meshL  = 3 // L-stack depth for StepMesh / StepTween
)

// needsLStack is true for MeshTween / MeshTweenChain: step.Forward needs dense
// layers along L (test41 mesh layout). PlaceStack-of-Stack in one cell returns
// nil mesh output under StepMesh.
func needsMeshVolume(mode parallel.TrainMode) bool {
	return mode == parallel.ModeMeshTween
}

func needsLStack(mode parallel.TrainMode) bool {
	return mode == parallel.ModeMeshTweenChain
}

func buildMLP(rng *rand.Rand) (*parallel.Stack, error) {
	mk := func(in, out int, act core.ActivationType) (*dense.Layer, error) {
		scale := float32(1 / math.Sqrt(float64(in)))
		w := make([]float32, in*out)
		for i := range w {
			w[i] = (rng.Float32()*2 - 1) * scale
		}
		return dense.NewConfigured(in, out, act, core.DTypeFloat32, quant.FormatNone, w)
	}
	a, err := mk(inDim, hidDim, core.ActivationTanh)
	if err != nil {
		return nil, err
	}
	b, err := mk(hidDim, hidDim, core.ActivationTanh)
	if err != nil {
		return nil, err
	}
	c, err := mk(hidDim, outDim, core.ActivationLinear)
	if err != nil {
		return nil, err
	}
	st, err := parallel.NewStack(a, b, c)
	if err != nil {
		return nil, err
	}
	st.Exec.Backend = core.BackendCPUTiled
	st.Exec.MultiCore = true
	st.SyncChildExec()
	return st, nil
}

// placeOn3x3x3 puts a Stack at origin on a 3×3×3 cube; other cells stay disabled.
func placeOn3x3x3(st *parallel.Stack) (*architecture.Grid, error) {
	g := architecture.NewGrid(gridN, gridN, gridN, 1)
	for i := range g.Cells {
		g.Cells[i].Layer.IsDisabled = true
	}
	if err := parallel.PlaceStack(g, 0, 0, 0, 0, st); err != nil {
		return nil, err
	}
	if c := g.At(0, 0, 0, 0); c != nil {
		c.Layer.IsDisabled = false
	}
	g.Exec = st.Exec
	return g, nil
}


// buildMeshVolume3 fills every cell of a 3×3×3×1 grid with hidDim→hidDim dense
// (test41-style uniform width). StepMesh/ApplyTween need a compact live volume —
// a single origin column on 3³ leaves ApplyTween inert (Δw=0).
func buildMeshVolume3(rng *rand.Rand) (*architecture.Grid, []*dense.Layer, error) {
	g := architecture.NewGrid(gridN, gridN, gridN, 1)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = true
	mk := func() (*dense.Layer, error) {
		scale := float32(1 / math.Sqrt(float64(hidDim)))
		w := make([]float32, hidDim*hidDim)
		for i := range w {
			w[i] = (rng.Float32()*2 - 1) * scale
		}
		return dense.NewConfigured(hidDim, hidDim, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, w)
	}
	layers := make([]*dense.Layer, 0, gridN*gridN*gridN)
	for z := 0; z < gridN; z++ {
		for y := 0; y < gridN; y++ {
			for x := 0; x < gridN; x++ {
				d, err := mk()
				if err != nil {
					return nil, nil, err
				}
				if err := dense.Place(g, z, y, x, 0, d); err != nil {
					return nil, nil, err
				}
				if c := g.At(z, y, x, 0); c != nil {
					c.Layer.IsDisabled = false
				}
				layers = append(layers, d)
			}
		}
	}
	return g, layers, nil
}

// buildMeshLStack3 places 3 uniform dens along L at origin of a 3×3×3×3 grid
// (used by MeshTweenChain / sequential Forward). Other cells stay disabled.
func buildMeshLStack3(rng *rand.Rand) (*architecture.Grid, []*dense.Layer, error) {
	g := architecture.NewGrid(gridN, gridN, gridN, meshL)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = true
	for i := range g.Cells {
		g.Cells[i].Layer.IsDisabled = true
	}
	mk := func(act core.ActivationType) (*dense.Layer, error) {
		scale := float32(1 / math.Sqrt(float64(hidDim)))
		w := make([]float32, hidDim*hidDim)
		for i := range w {
			w[i] = (rng.Float32()*2 - 1) * scale
		}
		return dense.NewConfigured(hidDim, hidDim, act, core.DTypeFloat32, quant.FormatNone, w)
	}
	acts := []core.ActivationType{core.ActivationTanh, core.ActivationTanh, core.ActivationLinear}
	layers := make([]*dense.Layer, 0, meshL)
	for l, act := range acts {
		d, err := mk(act)
		if err != nil {
			return nil, nil, err
		}
		if err := dense.Place(g, 0, 0, 0, l, d); err != nil {
			return nil, nil, err
		}
		if c := g.At(0, 0, 0, l); c != nil {
			c.Layer.IsDisabled = false
		}
		layers = append(layers, d)
	}
	return g, layers, nil
}


// padXOR lifts 2-bit XOR into hidDim vectors so StepMesh L-stacks (uniform width) can train.
func padXOR(xs, ys []*core.Tensor[float32]) (px, py []*core.Tensor[float32]) {
	px = make([]*core.Tensor[float32], len(xs))
	py = make([]*core.Tensor[float32], len(ys))
	for i := range xs {
		x := core.NewTensor[float32](1, hidDim)
		if len(xs[i].Data) >= 2 {
			x.Data[0], x.Data[1] = xs[i].Data[0], xs[i].Data[1]
		}
		y := core.NewTensor[float32](1, hidDim)
		if len(ys[i].Data) >= 1 {
			y.Data[0] = ys[i].Data[0]
		}
		px[i], py[i] = x, y
	}
	return px, py
}

func xorBatch(rng *rand.Rand, n int) (xs, ys []*core.Tensor[float32]) {
	xs = make([]*core.Tensor[float32], n)
	ys = make([]*core.Tensor[float32], n)
	for i := 0; i < n; i++ {
		a := float32(rng.Intn(2))
		b := float32(rng.Intn(2))
		x := core.NewTensor[float32](1, inDim)
		x.Data[0], x.Data[1] = a, b
		y := core.NewTensor[float32](1, outDim)
		y.Data[0] = float32(int(a) ^ int(b))
		xs[i], ys[i] = x, y
	}
	return xs, ys
}

func evalMSEAcc(st *parallel.Stack, grid *architecture.Grid, mode parallel.TrainMode, xs, ys []*core.Tensor[float32]) (mse, hardAcc float64, err error) {
	var sumMSE float64
	var correct, total int
	for i := range xs {
		post, ferr := infer(st, grid, mode, xs[i])
		if ferr != nil {
			return 0, 0, ferr
		}
		if post == nil || len(post.Data) < 1 || len(ys[i].Data) < 1 {
			return 0, 0, fmt.Errorf("nil post")
		}
		d := float64(post.Data[0] - ys[i].Data[0])
		sumMSE += d * d
		pred := 0
		if post.Data[0] >= 0.5 {
			pred = 1
		}
		want := 0
		if ys[i].Data[0] >= 0.5 {
			want = 1
		}
		if pred == want {
			correct++
		}
		total++
	}
	if total == 0 {
		return 0, 0, fmt.Errorf("empty eval")
	}
	return sumMSE / float64(total), 100 * float64(correct) / float64(total), nil
}

func infer(st *parallel.Stack, grid *architecture.Grid, mode parallel.TrainMode, x *core.Tensor[float32]) (*core.Tensor[float32], error) {
	switch {
	case mode == parallel.ModeMeshBP || mode == parallel.ModeMeshTween || mode == parallel.ModeMeshTweenChain:
		if grid == nil {
			return nil, fmt.Errorf("mesh infer needs grid")
		}
		fwd, err := forward.Forward(grid, x)
		if err != nil || fwd == nil {
			return nil, err
		}
		return fwd.Output, nil
	case mode.IsSplitFamily():
		tape, err := parallel.OpenSplitTape(st, x)
		if err != nil || tape == nil {
			return nil, err
		}
		return tape.Post, nil
	default:
		_, p, err := parallel.ForwardStack(st, x)
		return p, err
	}
}

func trainOne(
	st *parallel.Stack,
	grid *architecture.Grid,
	mode parallel.TrainMode,
	x, y *core.Tensor[float32],
	lr float64,
) error {
	switch mode {
	case parallel.ModeMeshBP:
		fwd, err := forward.Forward(grid, x)
		if err != nil || fwd == nil {
			return err
		}
		_, err = training.Step(fwd, y, lr)
		return err
	case parallel.ModeMeshTween:
		// Full 3×3×3 live volume: last cell is live, stock StepMesh is fine.
		// ticks ≥ cell count so signal reaches the end of the linear idx chain.
		ticks := len(grid.Cells)
		if ticks < 1 {
			ticks = gridN * gridN * gridN
		}
		_, _, err := training.StepMesh(grid, x, y, ticks, lr)
		return err
	case parallel.ModeMeshTweenChain:
		_, _, err := training.StepTween(grid, x, y, lr)
		return err
	}
	if mode.IsSplitFamily() {
		tape, err := parallel.OpenSplitTape(st, x)
		if err != nil || tape == nil {
			return err
		}
		_, err = tape.Train(y, mode, lr)
		return err
	}
	_, err := parallel.TrainStackMSE(st, x, y, mode, lr)
	return err
}


// trainStepMeshOrigin runs step ticks on a 3×3×3×L grid and ApplyTween using the
// live origin column (training.StepMesh wrongly takes the last cell in the slab).
func trainStepMeshOrigin(grid *architecture.Grid, x, y *core.Tensor[float32], ticks int, lr float64) error {
	if grid == nil || x == nil || y == nil {
		return fmt.Errorf("trainStepMeshOrigin nil")
	}
	if ticks < 1 {
		ticks = meshL
	}
	st := step.New[float32](grid)
	st.SetInput(x)
	for t := 0; t < ticks; t++ {
		if _, err := step.Forward(grid, st, t == ticks-1); err != nil {
			return err
		}
	}
	outIdx := grid.Index(0, 0, 0, meshL-1)
	if outIdx < 0 || outIdx >= len(st.LayerData) || st.LayerData[outIdx] == nil {
		return fmt.Errorf("nil mesh out at origin L=%d idx=%d", meshL-1, outIdx)
	}
	return step.ApplyTween(grid, st, y, float32(lr))
}

func weightNorm(st *parallel.Stack) float64 {
	if st == nil {
		return 0
	}
	return weightNormOp(st)
}

func weightNormOp(op any) float64 {
	var sum float64
	for _, s := range dna.CollectStores(op) {
		if s == nil {
			continue
		}
		f, err := s.FlattenF32()
		if err != nil {
			continue
		}
		for _, v := range f {
			sum += float64(v) * float64(v)
		}
	}
	return math.Sqrt(sum)
}

func weightNormLayers(layers []*dense.Layer) float64 {
	var sum float64
	for _, l := range layers {
		n := weightNormOp(l)
		sum += n * n
	}
	return math.Sqrt(sum)
}

func weightDelta(a, b float64) float64 {
	return math.Abs(a - b)
}
