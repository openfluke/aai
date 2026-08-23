package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
)

type row struct {
	Mode      string
	Loss0     float64
	Loss1     float64
	Acc0      float64
	Acc1      float64
	WDelta    float64
	Steps     int64
	Err       string
	Pass      bool
	RequiresG bool
	Layout    string // "stack@3³" or "L-stack@3³"
}

func main() {
	modesFlag := flag.String("modes", "all", "all = AllNamedTrainModes, or csv")
	dur := flag.Duration("duration", 1500*time.Millisecond, "wall train per mode")
	lr := flag.Float64("lr", 0.05, "learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	evalN := flag.Int("eval", 64, "eval XOR samples")
	flag.Parse()

	modes, err := parseModes(*modesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  test52 — all train modes on 3×3×3 mesh (XOR learn smoke)        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Printf("grid=%dx%dx%d  modes=%d  wall=%s  lr=%g  seed=%d\n",
		gridN, gridN, gridN, len(modes), *dur, *lr, *seed)
	fmt.Println("MeshTween/MeshTweenChain use L-stacked dense on 3³ (StepMesh needs it).")
	fmt.Println()

	rows := make([]row, 0, len(modes))
	fail := 0
	for i, mode := range modes {
		r := smokeMode(mode, *seed+int64(i)*97, *dur, *lr, *evalN)
		rows = append(rows, r)
		mark := "PASS"
		if !r.Pass {
			mark = "FAIL"
			fail++
		}
		extra := ""
		if r.RequiresG {
			extra = " [grid]"
		}
		if r.Err != "" {
			fmt.Printf("[%2d/%d] %-28s %s%s  err=%s\n", i+1, len(modes), r.Mode, mark, extra, r.Err)
		} else {
			fmt.Printf("[%2d/%d] %-28s %s%s  loss %.4f→%.4f  Acc %.0f→%.0f  |Δw|=%.3g  steps=%d  (%s)\n",
				i+1, len(modes), r.Mode, mark, extra,
				r.Loss0, r.Loss1, r.Acc0, r.Acc1, r.WDelta, r.Steps, r.Layout)
		}
	}

	fmt.Println()
	fmt.Println("── summary ──")
	fmt.Printf("%-28s %6s %10s %10s %8s %8s %s\n",
		"mode", "result", "loss0", "loss1", "Acc0", "Acc1", "note")
	for _, r := range rows {
		mark := "PASS"
		note := r.Layout
		if !r.Pass {
			mark = "FAIL"
			note = r.Err
			if note == "" {
				note = "no learn signal"
			}
		} else if r.RequiresG {
			note = r.Layout + " RequiresGrid"
		}
		fmt.Printf("%-28s %6s %10.4f %10.4f %7.1f %7.1f %s\n",
			r.Mode, mark, r.Loss0, r.Loss1, r.Acc0, r.Acc1, note)
	}
	fmt.Printf("\n%d / %d PASS on 3×3×3\n", len(rows)-fail, len(rows))
	if fail > 0 {
		os.Exit(1)
	}
}

func smokeMode(mode parallel.TrainMode, seed int64, dur time.Duration, lr float64, evalN int) row {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	r := row{Mode: mode.String(), RequiresG: mode.RequiresGrid()}
	rng := rand.New(rand.NewSource(seed))

	var st *parallel.Stack
	var grid *architecture.Grid
	var dens []*dense.Layer
	var err error

	if needsMeshVolume(mode) {
		r.Layout = "volume@3³"
		grid, dens, err = buildMeshVolume3(rng)
		if err != nil {
			r.Err = "mesh volume: " + err.Error()
			return r
		}
	} else if needsLStack(mode) {
		r.Layout = "L-stack@3³"
		grid, dens, err = buildMeshLStack3(rng)
		if err != nil {
			r.Err = "mesh L-stack: " + err.Error()
			return r
		}
	} else {
		r.Layout = "stack@3³"
		st, err = buildMLP(rng)
		if err != nil {
			r.Err = "build: " + err.Error()
			return r
		}
		grid, err = placeOn3x3x3(st)
		if err != nil {
			r.Err = "place 3×3×3: " + err.Error()
			return r
		}
	}
	if dens == nil && (grid.Depth != gridN || grid.Rows != gridN || grid.Cols != gridN) {
		r.Err = fmt.Sprintf("grid shape %d×%d×%d want %d³", grid.Depth, grid.Rows, grid.Cols, gridN)
		return r
	}

	evalX, evalY := xorBatch(rand.New(rand.NewSource(seed+1)), evalN)
	if dens != nil {
		evalX, evalY = padXOR(evalX, evalY)
	}
	loss0, acc0, err := evalMSEAcc(st, grid, mode, evalX, evalY)
	if err != nil {
		r.Err = "eval0: " + err.Error()
		return r
	}
	r.Loss0, r.Acc0 = loss0, acc0
	w0 := 0.0
	if dens != nil {
		w0 = weightNormLayers(dens)
	} else {
		w0 = weightNorm(st)
	}

	trainX, trainY := xorBatch(rng, 4096)
	if dens != nil {
		trainX, trainY = padXOR(trainX, trainY)
	}
	deadline := time.Now().Add(dur)
	var steps int64
	ti := 0
	for time.Now().Before(deadline) {
		x, y := trainX[ti%len(trainX)], trainY[ti%len(trainY)]
		ti++
		if err := trainOne(st, grid, mode, x, y, lr); err != nil {
			r.Err = "train: " + err.Error()
			r.Steps = steps
			return r
		}
		steps++
	}
	r.Steps = steps

	loss1, acc1, err := evalMSEAcc(st, grid, mode, evalX, evalY)
	if err != nil {
		r.Err = "eval1: " + err.Error()
		return r
	}
	r.Loss1, r.Acc1 = loss1, acc1
	w1 := 0.0
	if dens != nil {
		w1 = weightNormLayers(dens)
	} else {
		w1 = weightNorm(st)
	}
	r.WDelta = weightDelta(w0, w1)

	if math.IsNaN(r.Loss1) || math.IsInf(r.Loss1, 0) {
		r.Err = "NaN/Inf loss"
		return r
	}

	lossDrop := r.Loss1 < r.Loss0-1e-4
	accUp := r.Acc1 > r.Acc0+0.5
	moved := r.WDelta > 1e-6
	r.Pass = (lossDrop || accUp) && moved
	if !r.Pass && moved && r.Loss1 <= r.Loss0+0.05 {
		r.Pass = r.Acc1 >= 50 || lossDrop
	}
	return r
}

func parseModes(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return parallel.AllNamedTrainModes(), nil
	}
	var out []parallel.TrainMode
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		m, err := parallel.ParseTrainMode(tok)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no modes in %q", spec)
	}
	return out, nil
}
