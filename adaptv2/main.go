package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/simd"
)

type result struct {
	Mode    string        `json:"mode"`
	SIMD    bool          `json:"simd"`
	Acc1s   []float64     `json:"acc_1s"`
	Soft1s  []float64     `json:"soft_1s"`
	Lucy    lucy.Snapshot `json:"lucy"`
	RAMKiB  float64       `json:"ram_kib"`
	Outputs int64         `json:"outputs"`
	Err     string        `json:"error,omitempty"`
}

func main() {
	dur := flag.Duration("duration", 15*time.Second, "wall per mode (loom [2] is 15s)")
	phase := flag.Duration("phase", 5*time.Second, "chase / avoid / chase slice")
	win := flag.Duration("window", time.Second, "accuracy bucket")
	lr := flag.Float64("lr", 0.05, "learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	modesFlag := flag.String("modes", "all", "all | named | loom | step | credit | csv of ParseTrainMode names")
	simdFlag := flag.String("simd", "off", "off | on | both")
	workers := flag.Int("workers", 1, "concurrent mode runs (1 = Lucy-honest Avail)")
	outJSON := flag.String("json", "adaptv2_results.json", "results path (empty = skip)")
	flag.Parse()

	modes, err := parseModeList(*modesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	backends, err := parseSIMD(*simdFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	nWin := int(*dur / *win)
	if nWin < 1 {
		nWin = 1
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  adaptv2 — dense mid-stream adaptation (Welvet × Lucy density)           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Timeline: [Chase %s] → [AVOID %s] → [Chase %s]\n", *phase, *phase, *phase)
	fmt.Println("Network:  6-layer Dense (8→32→64→64→64→32→4)  — loom lucy_bloom_rivers [2]")
	fmt.Printf("Engine:   Welvet Stack  duty=%s  SIMD linked=%v\n", dutyClockName(), simd.Enabled())
	fmt.Println(parallel.ShortTrainModeLegend)
	fmt.Printf("Modes:    %d   backends: %d   workers: %d   wall/job: %s\n\n", len(modes), len(backends), *workers, *dur)

	type job struct {
		mode parallel.TrainMode
		be   core.Backend
		simd bool
	}
	var jobs []job
	for _, be := range backends {
		for _, m := range modes {
			jobs = append(jobs, job{mode: m, be: be, simd: be == core.BackendSIMD})
		}
	}

	results := make([]result, len(jobs))
	sem := make(chan struct{}, max(1, *workers))
	var wg sync.WaitGroup
	for i, j := range jobs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, j job) {
			defer wg.Done()
			defer func() { <-sem }()
			tag := j.mode.Short()
			if j.simd {
				tag += " SIMD"
			}
			fmt.Printf("Starting [%s]...\n", tag)
			results[i] = runJob(j.mode, j.be, *dur, *phase, *win, nWin, *lr, *seed+int64(i)*17)
			if results[i].Err != "" {
				fmt.Printf("Finished [%s] — ERR %s\n", tag, results[i].Err)
				return
			}
			fmt.Printf("Finished [%s] — Total outputs: %d\n", tag, results[i].Outputs)
		}(i, j)
	}
	wg.Wait()
	fmt.Print("\nAll tests complete.\n\n")

	printAccTable(results, nWin, *phase, *win)
	printAdaptSummary(results, nWin, *phase, *win)
	printOps(results)
	printLPD(results)

	if strings.TrimSpace(*outJSON) != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := os.WriteFile(*outJSON, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("💾 %s\n", *outJSON)
	}
}

func parseModeList(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return parallel.AllStackLocalTrainModes(), nil
	}
	if strings.EqualFold(spec, "named") {
		return parallel.AllNamedTrainModes(), nil
	}
	if strings.EqualFold(spec, "loom") {
		return []parallel.TrainMode{
			parallel.ModeNormalBP,
			parallel.ModeStepBP,
			parallel.ModeTween,
			parallel.ModeTweenChain,
			parallel.ModeStepTween,
			parallel.ModeStepTweenChain,
		}, nil
	}
	if strings.EqualFold(spec, "credit") {
		return parallel.AllCreditTrainModes(), nil
	}
	if strings.EqualFold(spec, "step") || strings.EqualFold(spec, "step*") {
		var out []parallel.TrainMode
		for _, m := range parallel.AllStackLocalTrainModes() {
			if m.IsLineStep() {
				out = append(out, m)
			}
		}
		return out, nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		m, err := parallel.ParseTrainMode(tok)
		if err != nil {
			return nil, err
		}
		if m == parallel.ModeInherit || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no train modes in %q", spec)
	}
	return out, nil
}

func parseSIMD(spec string) ([]core.Backend, error) {
	switch strings.ToLower(strings.TrimSpace(spec)) {
	case "off", "cpu", "scalar":
		return []core.Backend{core.BackendCPUTiled}, nil
	case "on", "simd":
		if !simd.Enabled() {
			return nil, fmt.Errorf("SIMD not linked")
		}
		return []core.Backend{core.BackendSIMD}, nil
	case "both":
		out := []core.Backend{core.BackendCPUTiled}
		if simd.Enabled() {
			out = append(out, core.BackendSIMD)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unknown -simd %q (off|on|both)", spec)
	}
}

func runJob(mode parallel.TrainMode, be core.Backend, dur, phase, win time.Duration, nWin int, lr float64, seed int64) result {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	r := result{Mode: mode.String(), SIMD: be == core.BackendSIMD, Acc1s: make([]float64, nWin), Soft1s: make([]float64, nWin)}
	rng := rand.New(rand.NewSource(seed))
	teach := newTeacher(rng)
	st, err := buildChaseNet(rng, be)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.Lucy.WeightBytes = stackWeightBytes(st)
	r.RAMKiB = float64(r.Lucy.WeightBytes) / 1024

	var grid *architecture.Grid
	if mode.RequiresGrid() {
		g, gerr := placeForMesh(st)
		if gerr != nil {
			r.Err = "place: " + gerr.Error()
			return r
		}
		grid = g
	}

	wins := make([]lucy.Window, nWin)
	var infSum, trSum time.Duration
	start := time.Now()
	prevPhase := ""

	for time.Since(start) < dur {
		elapsed := time.Since(start)
		wi := int(elapsed / win)
		if wi >= nWin {
			wi = nWin - 1
		}
		avoid := false
		ph := "chase1"
		switch {
		case elapsed >= 2*phase:
			ph = "chase2"
		case elapsed >= phase:
			avoid = true
			ph = "avoid"
		}
		if prevPhase != "" && ph != prevPhase {
			wins[wi].PhaseSwitches++
		}
		prevPhase = ph
		wins[wi].Phase = ph

		x := sampleX(rng)
		y := labelFor(teach, x, avoid)

		tInf := startWork()
		var post *core.Tensor[float32]
		var tape *parallel.SplitTape[float32]
		var meshFwd *forward.Result[float32]
		switch {
		case mode == parallel.ModeMeshBP || mode == parallel.ModeMeshTween || mode == parallel.ModeMeshTweenChain:
			if grid == nil {
				continue
			}
			fwd, ferr := forward.Forward(grid, x)
			if ferr != nil || fwd == nil || fwd.Output == nil {
				continue
			}
			meshFwd = fwd
			post = fwd.Output
		case mode.IsSplitFamily():
			tp, terr := parallel.OpenSplitTape(st, x)
			if terr != nil || tp == nil || tp.Post == nil {
				continue
			}
			tape = tp
			post = tp.Post
		default:
			_, p, ferr := parallel.ForwardStack(st, x)
			if ferr != nil || p == nil {
				continue
			}
			post = p
		}
		inf := tInf.elapsed()
		infSum += inf

		lab := argmax(y.Data)
		pred := argmax(post.Data)
		hard := 0.0
		if pred == lab {
			hard = 100
		}
		soft := lucy.SoftAccProb(softmaxPTrue(post.Data, lab), 1)

		r.Lucy.TotalOutputs++
		r.Lucy.TotalTrain++
		w := &wins[wi]
		w.Outputs++
		w.TrainSteps++
		w.InferMs += inf.Seconds() * 1000
		w.Accuracy += hard
		w.SoftAcc += soft
		if pred == lab {
			w.Correct++
			r.Lucy.TotalCorrect++
		}

		tTr := startWork()
		trainSample(st, grid, meshFwd, tape, x, y, mode, lr)
		tr := tTr.elapsed()
		trSum += tr
		w.TrainMs += tr.Seconds() * 1000
	}

	for i := range wins {
		if wins[i].Outputs > 0 {
			wins[i].Accuracy /= float64(wins[i].Outputs)
			wins[i].SoftAcc /= float64(wins[i].Outputs)
		}
		r.Acc1s[i] = wins[i].Accuracy
		r.Soft1s[i] = wins[i].SoftAcc
	}
	r.Lucy.Windows = wins
	r.Lucy.Duration = time.Since(start)
	r.Lucy.InferMs = infSum.Seconds() * 1000
	r.Lucy.TrainMs = trSum.Seconds() * 1000
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: 1, ConsThreshold: lucy.ConsThreshold})
	r.Outputs = r.Lucy.TotalOutputs
	return r
}
