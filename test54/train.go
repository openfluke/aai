package main

import (
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
)

// Fixed 1³ placement for Mesh* modes — no grid sweep.
func placeOrigin1(st *parallel.Stack) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
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

func trainSample(
	st *parallel.Stack,
	grid *architecture.Grid,
	meshFwd *forward.Result[float32],
	tape *parallel.SplitTape[float32],
	x, y *core.Tensor[float32],
	mode parallel.TrainMode,
	lr float64,
) {
	switch mode {
	case parallel.ModeMeshBP:
		fwd := meshFwd
		if fwd == nil && grid != nil {
			var err error
			fwd, err = forward.Forward(grid, x)
			if err != nil || fwd == nil {
				return
			}
		}
		if fwd != nil {
			_, _ = training.Step(fwd, y, lr)
		}
		return
	case parallel.ModeMeshTween:
		if grid != nil {
			_, _, _ = training.StepMesh(grid, x, y, 1, lr)
		}
		return
	case parallel.ModeMeshTweenChain:
		if grid != nil {
			_, _, _ = training.StepTween(grid, x, y, lr)
		}
		return
	}
	if tape != nil {
		_, _ = tape.Train(y, mode, lr)
		return
	}
	_, _ = parallel.TrainStackMSE(st, x, y, mode, lr)
}
