package main

import (
	"flag"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/runtime/forward"
)

const (
	windowDuration = 50 * time.Millisecond
	defaultHidden  = 32
)

// Default mode list — every named update the user listed (Mesh* on fixed 1³).
var defaultModesCSV = strings.Join([]string{
	"sgd", "step_sgd", "tween", "tween_chain", "step_tween", "step_tween_chain",
	"MeshBP", "MeshTween", "MeshTweenChain", "TweenSplit", "StepTweenSplit",
	"TweenAlt", "StepTweenAlt", "TweenSplitHeadProxy", "TweenSplitLinear",
	"TweenSplitFastProxy", "TweenSplitLinearCache", "TweenSplitHeadProxyAsync",
	"TweenSplitSparse", "MeshTweenSplit", "MeshTweenAlt", "MeshTweenSplitFastProxy",
	"MeshTweenSplitSparse", "StepTweenSplitHeadProxy", "StepTweenSplitLinear",
	"StepTweenSplitFastProxy", "StepTweenSplitLinearCache",
	"StepTweenSplitHeadProxyAsync", "StepTweenSplitSparse",
}, ",")

type Job struct {
	ID    string
	Kind  CellKind
	DType core.DType
	Mode  parallel.TrainMode
}

type modeResult struct {
	ID      string        `json:"id"`
	Layer   string        `json:"layer"`
	DType   string        `json:"dtype"`
	Mode    string        `json:"mode"`
	Acc     float64       `json:"acc"`
	Soft    float64       `json:"soft"`
	Avail   float64       `json:"avail"`
	Thru    float64       `json:"thru"`
	Score   float64       `json:"score"`
	Adapt   float64       `json:"adapt"`
	Days    float64       `json:"days_done"` // closed-loop days completed in eval
	RAMKiB  float64       `json:"ram_kib"`
	Actions int64         `json:"actions"`
	Err     string        `json:"err,omitempty"`
	Lucy    lucy.Snapshot `json:"lucy"`
}

func main() {
	LoadDotEnv(".env")

	modesFlag := flag.String("modes", EnvOr("TEST53_MODES", defaultModesCSV), "csv of train modes")
	layersFlag := flag.String("layers", EnvOr("TEST53_LAYERS", "all"), "all|csv cell kinds")
	dtypesFlag := flag.String("dtypes", EnvOr("TEST53_DTYPES", "all"), "all|csv")
	workersFlag := flag.Int("workers", EnvInt("TEST53_WORKERS", 0), "0 = NumCPU")
	dur := flag.Duration("duration", EnvDuration("TEST53_DURATION", 2*time.Second), "wall per job")
	lr := flag.Float64("lr", EnvFloat("TEST53_LR", 0.05), "learning rate")
	hidden := flag.Int("hidden", EnvInt("TEST53_HIDDEN", defaultHidden), "sandwich hidden")
	ckpt := flag.String("ckpt", EnvOr("TEST53_CKPT", "test53_ckpt"), "checkpoint dir")
	resume := flag.Bool("resume", EnvBool("TEST53_RESUME", true), "skip done IDs")
	tideAddr := flag.String("tide-addr", EnvOr("TEST53_TIDE_ADDR", "0.0.0.0:8080"), "Tide dash (empty=off)")
	flag.Parse()

	modes, err := parseModeList(*modesFlag)
	must(err)
	kinds, err := parseLayerList("dense", *layersFlag, nil)
	must(err)
	dtypes, err := parseDTypeList(*dtypesFlag)
	must(err)

	workers := *workersFlag
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers < 1 {
		workers = 1
	}
	leafMultiCore = workers == 1

	jobs := expandJobs(kinds, modes, dtypes)
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  test53 · dayroute (5-day life) × layer × mode × dtype      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("task=dayroute  grid=%dx%d  days=%d  acts=%d  schedule=%v\n",
		gridSize, gridSize, nDays, nActs, []string{"wake", "bath", "breakfast", "work", "lunch", "gym", "couch", "sleep"})
	fmt.Printf("kinds=%d modes=%d dtypes=%d jobs=%d workers=%d leafMultiCore=%v dur=%s lr=%g\n",
		len(kinds), len(modes), len(dtypes), len(jobs), workers, leafMultiCore, *dur, *lr)
	fmt.Printf("kinds=%s\n", kindsCSV(kinds))
	fmt.Printf("clock=%s  ckpt=%s\n\n", dutyClockName(), *ckpt)

	store := NewStore(*ckpt)
	prog, err := store.Load()
	must(err)
	done := doneSet(prog.DoneIDs)
	if !*resume {
		done = map[string]bool{}
		prog = &Progress{}
	}
	prog.Total = len(jobs)

	prior, _ := store.LoadResults()
	results := make([]modeResult, 0, len(prior)+len(jobs))
	results = append(results, prior...)
	for _, r := range prior {
		done[r.ID] = true
	}

	pending := make([]Job, 0, len(jobs))
	for _, j := range jobs {
		if !done[j.ID] {
			pending = append(pending, j)
		}
	}
	fmt.Printf("resume: done=%d pending=%d\n", len(jobs)-len(pending), len(pending))

	tide := startTideBridge(*tideAddr, jobs, *lr)
	alreadyDone := len(jobs) - len(pending)
	if tide != nil {
		tide.seedCompleted(results)
		tide.setQueue(alreadyDone, len(jobs), fmt.Sprintf("ready · %d/%d · %d left", alreadyDone, len(jobs), len(pending)))
		tide.signalStart()
	}

	if len(pending) == 0 {
		fmt.Println("nothing pending — rebuilding LPD from ckpt")
		writeLPD(store, results)
		printTopLPD(results, 15)
		if tide != nil {
			fmt.Println("tide still serving; Ctrl-C to exit")
			select {}
		}
		return
	}

	type leafOut struct {
		job Job
		res modeResult
	}
	jobCh := make(chan Job, workers*2)
	outCh := make(chan leafOut, workers*2)
	var wg sync.WaitGroup
	var started atomic.Int64
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			for j := range jobCh {
				n := int(started.Add(1))
				if tide != nil {
					// done-so-far for display = alreadyDone + finished; starting count is approximate
					tide.beginJob(j, "train", alreadyDone+n-1, len(jobs))
				}
				outCh <- leafOut{job: j, res: runJob(j, seed, *hidden, *dur, *lr)}
			}
		}(int64(w+1) * 9973)
	}
	go func() {
		for _, j := range pending {
			jobCh <- j
		}
		close(jobCh)
		wg.Wait()
		close(outCh)
	}()

	var finished int64
	totalPending := len(pending)
	planTotal := len(jobs)
	for out := range outCh {
		r := out.res
		j := out.job
		n := int(atomic.AddInt64(&finished, 1))
		doneNow := alreadyDone + n
		if tide != nil {
			// Re-begin with the real job (mode!) then finish so mode_progress.Left ticks down.
			tide.beginJob(j, "train", doneNow-1, planTotal)
			tide.pulseRunning(r)
			tide.finishJob(r)
			left := planTotal - doneNow
			tide.setQueue(doneNow, planTotal, fmt.Sprintf("%s · %d/%d · %d left", r.ID, doneNow, planTotal, left))
		}
		results = append(results, r)
		prog.DoneIDs = append(prog.DoneIDs, r.ID)
		prog.NextIndex = len(prog.DoneIDs)
		prog.Current = r.ID
		if r.Err == "" && (prog.BestID == "" || r.Score > prog.BestScore || (r.Score == prog.BestScore && r.Acc > prog.BestAcc)) {
			prog.BestID, prog.BestScore, prog.BestAcc = r.ID, r.Score, r.Acc
		}
		_ = store.SaveProgress(prog)
		_ = store.AppendHistory(map[string]any{
			"at": time.Now().UTC(), "id": r.ID, "acc": r.Acc, "score": r.Score,
			"avail": r.Avail, "err": r.Err,
		})
		_ = store.SaveJSON("results.json", map[string]any{
			"results": results, "best_id": prog.BestID, "best_score": prog.BestScore,
			"done": doneNow, "total": planTotal, "left": planTotal - doneNow,
		})
		if n%25 == 0 || n == totalPending || r.Err != "" {
			tag := "ok"
			if r.Err != "" {
				tag = "ERR " + trim(r.Err, 60)
			}
			fmt.Printf("  [%d/%d pending · %d left] Acc %.1f Score %.0f · %s · %s\n",
				n, totalPending, planTotal-doneNow, r.Acc, r.Score, r.ID, tag)
		}
		if n%100 == 0 || n == totalPending {
			writeLPD(store, results)
		}
	}

	writeLPD(store, results)
	printTopLPD(results, 20)
	if tide != nil {
		fmt.Println("done — tide still serving; Ctrl-C to exit")
		select {}
	}
	fmt.Println("done")
}

func expandJobs(kinds []CellKind, modes []parallel.TrainMode, dtypes []core.DType) []Job {
	out := make([]Job, 0, len(kinds)*len(modes)*len(dtypes))
	for _, k := range kinds {
		for _, m := range modes {
			for _, dt := range dtypes {
				id := fmt.Sprintf("%s|%s|%s", k, dt, m.String())
				out = append(out, Job{ID: id, Kind: k, DType: dt, Mode: m})
			}
		}
	}
	return out
}

func parseModeList(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return parallel.AllNamedTrainModes(), nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, tok := range strings.FieldsFunc(spec, func(r rune) bool {
		return r == ',' || r == ' ' || r == ';'
	}) {
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
		return nil, fmt.Errorf("no modes in %q", spec)
	}
	return out, nil
}

func runJob(j Job, seed int64, hidden int, dur time.Duration, lr float64) modeResult {
	r := modeResult{
		ID: j.ID, Layer: string(j.Kind), DType: j.DType.String(), Mode: j.Mode.String(),
	}
	st, err := buildSandwich(j.Kind, obsDim, hidden, nActs, j.DType)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.RAMKiB = float64(stackWeightBytes(st)) / 1024
	r.Lucy.WeightBytes = stackWeightBytes(st)

	var grid *architecture.Grid
	if j.Mode.RequiresGrid() {
		grid, err = placeOrigin1(st)
		if err != nil {
			r.Err = "place: " + err.Error()
			return r
		}
	}

	world := newDayWorld(seed ^ int64(len(j.ID))*131)
	numWindows := int(dur / windowDuration)
	if numWindows < 1 {
		numWindows = 1
	}
	win := make([]lucy.Window, numWindows)
	start := time.Now()
	var totalInfer, totalTrain time.Duration
	currentWindow := 0
	lastDay := world.doneDay

	for time.Since(start) < dur {
		elapsed := time.Since(start)
		nw := int(elapsed / windowDuration)
		if nw > currentWindow && nw < numWindows {
			currentWindow = nw
		}
		if world.doneDay > lastDay {
			lastDay = world.doneDay
			if currentWindow < numWindows {
				win[currentWindow].PhaseSwitches++
			}
		}
		s := world.advanceTeacher()

		tInf := startWork()
		var post *core.Tensor[float32]
		var tape *parallel.SplitTape[float32]
		var meshFwd *forward.Result[float32]
		switch {
		case j.Mode == parallel.ModeMeshBP || j.Mode == parallel.ModeMeshTween || j.Mode == parallel.ModeMeshTweenChain:
			fwd, ferr := forward.Forward(grid, s.x)
			if ferr != nil || fwd == nil || fwd.Output == nil {
				continue
			}
			meshFwd, post = fwd, fwd.Output
		case j.Mode.IsSplitFamily():
			tp, terr := parallel.OpenSplitTape(st, s.x)
			if terr != nil || tp == nil || tp.Post == nil {
				continue
			}
			tape, post = tp, tp.Post
		default:
			_, p, ferr := parallel.ForwardStack(st, s.x)
			if ferr != nil || p == nil {
				continue
			}
			post = p
		}
		inferDur := tInf.elapsed()
		totalInfer += inferDur

		hard, soft := scoreAct(post, s.y)
		r.Actions++
		if currentWindow < numWindows {
			w := &win[currentWindow]
			w.Outputs++
			w.InferMs += inferDur.Seconds() * 1000
			w.SoftAcc += soft
			w.Accuracy += hard
		}

		t0 := startWork()
		trainSample(st, grid, meshFwd, tape, s.x, s.y, j.Mode, lr)
		trainDur := t0.elapsed()
		if trainDur > 0 {
			totalTrain += trainDur
			if currentWindow < numWindows {
				win[currentWindow].TrainMs += trainDur.Seconds() * 1000
			}
		}
	}

	for i := range win {
		if win[i].Outputs > 0 {
			win[i].SoftAcc /= float64(win[i].Outputs)
			win[i].Accuracy /= float64(win[i].Outputs)
		}
	}
	r.Lucy.Windows = win
	r.Lucy.Duration = time.Since(start)
	r.Lucy.InferMs = totalInfer.Seconds() * 1000
	r.Lucy.TrainMs = totalTrain.Seconds() * 1000
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: 10, ConsThreshold: lucy.ConsThreshold})

	acc, soft, days := evalRoute(st, grid, j.Mode, seed+7, 256)
	r.Acc = acc
	r.Days = days
	r.Soft = soft
	if r.Lucy.SoftAcc > 0 {
		r.Soft = r.Lucy.SoftAcc
	}
	r.Avail = r.Lucy.Availability
	r.Thru = r.Lucy.Throughput
	r.Score = r.Lucy.Score
	r.Adapt = r.Lucy.AdaptPct
	return r
}

func writeLPD(store *Store, results []modeResult) {
	pts := make([]lucy.Sample, 0, len(results))
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		pts = append(pts, lucy.Sample{
			ID: r.ID, Mode: r.Mode, DType: r.DType, Arch: r.Layer,
			Score: r.Score, Soft: r.Soft, Acc: r.Acc, Thru: r.Thru, Avail: r.Avail, RAMKiB: r.RAMKiB,
		})
	}
	lpd := lucy.BuildLPD(pts)
	_ = store.SaveJSON("lpd.json", lpd)
}

func printTopLPD(results []modeResult, n int) {
	pts := make([]lucy.Sample, 0, len(results))
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		pts = append(pts, lucy.Sample{
			ID: r.ID, Mode: r.Mode, DType: r.DType, Arch: r.Layer,
			Score: r.Score, Soft: r.Soft, Acc: r.Acc, Thru: r.Thru, Avail: r.Avail, RAMKiB: r.RAMKiB,
		})
	}
	lpd := lucy.BuildLPD(pts)
	fmt.Println()
	fmt.Println("══ LPD top (layer × mode × dtype) ══")
	if lpd.Champ.ID != "" {
		fmt.Printf("score-champ  %s  Acc=%.1f Score=%.0f\n", lpd.Champ.ID, lpd.Champ.Acc, lpd.Champ.Score)
	}
	if lpd.LiveChamp.ID != "" {
		fmt.Printf("live-champ   %s  Acc=%.1f Score=%.0f\n", lpd.LiveChamp.ID, lpd.LiveChamp.Acc, lpd.LiveChamp.Score)
	}
	rows := append([]lucy.LPDRow(nil), lpd.Top...)
	if len(rows) == 0 {
		rows = append(rows, lpd.Gold...)
		rows = append(rows, lpd.Near...)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].LPD != rows[j].LPD {
			return rows[i].LPD > rows[j].LPD
		}
		return rows[i].Score > rows[j].Score
	})
	if n > len(rows) {
		n = len(rows)
	}
	fmt.Printf("%-48s %6s %6s %7s %8s\n", "id", "Acc", "Avail", "LPD", "Score")
	for i := 0; i < n; i++ {
		r := rows[i]
		fmt.Printf("%-48s %5.1f%% %5.1f%% %7.3f %8.0f\n", r.ID, r.Acc, r.Avail, r.LPD, r.Score)
	}
	ok, fail := 0, 0
	for _, r := range results {
		if r.Err != "" {
			fail++
		} else {
			ok++
		}
	}
	fmt.Printf("\ncompleted ok=%d fail=%d → test53_ckpt/lpd.json\n", ok, fail)
}

func trim(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
