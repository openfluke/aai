package main

import (
	"flag"
	"fmt"
	"runtime"
	"sort"
	"strconv"
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

// Default mode list (Mesh* on fixed 1³). Modes in remove_train_modes.md are omitted.
var defaultModesCSV = strings.Join([]string{
	"sgd", "step_sgd", "step_tween_chain",
	"MeshTween", "TweenSplit", "StepTweenSplit",
	"TweenSplitHeadProxy", "TweenSplitLinear",
	"TweenSplitFastProxy", "TweenSplitLinearCache", "TweenSplitHeadProxyAsync",
	"TweenSplitSparse", "MeshTweenSplit", "MeshTweenSplitFastProxy",
	"MeshTweenSplitSparse", "StepTweenSplitHeadProxy", "StepTweenSplitLinear",
	"StepTweenSplitFastProxy", "StepTweenSplitLinearCache",
	"StepTweenSplitHeadProxyAsync", "StepTweenSplitSparse",
}, ",")

// removedTrainModes — first-pass cut for temporal/3D search (see remove_train_modes.md).
var removedTrainModes = map[parallel.TrainMode]bool{
	parallel.ModeTween:          true, // tween / [T]
	parallel.ModeTweenChain:     true, // tween_chain / [T]Chain
	parallel.ModeStepTween:      true, // step_tween / Step[T]
	parallel.ModeTweenAlt:       true, // TweenAlt / [T]Alt
	parallel.ModeStepTweenAlt:   true, // StepTweenAlt / Step[T]Alt
	parallel.ModeMeshBP:         true, // MeshBP
	parallel.ModeMeshTweenAlt:   true, // MeshTweenAlt / Mesh[T]Alt
	parallel.ModeMeshTweenChain: true, // MeshTweenChain / Mesh[T]Chain
}
// Funny LR ramp: 0.02 → 2 → 200 → 2k → 20k → 1m → 10m → 100m
var defaultFunnyLRs = []float64{0.02, 2, 200, 2000, 20000, 1e6, 1e7, 1e8}

// Mild half (./run-docker-lo.sh).
var funnyLoLRs = []float64{0.02, 2, 200, 2000}

// Extreme half (./run-docker-hi.sh).
var funnyHiLRs = []float64{20000, 1e6, 1e7, 1e8}

func funnyLRs() []float64 {
	out := make([]float64, len(defaultFunnyLRs))
	copy(out, defaultFunnyLRs)
	return out
}

func funnyLo() []float64 {
	out := make([]float64, len(funnyLoLRs))
	copy(out, funnyLoLRs)
	return out
}

func funnyHi() []float64 {
	out := make([]float64, len(funnyHiLRs))
	copy(out, funnyHiLRs)
	return out
}

// Negated funny ramp (same magnitudes, sign flipped).
func funnyNegLRs() []float64 {
	out := make([]float64, len(defaultFunnyLRs))
	for i, v := range defaultFunnyLRs {
		out[i] = -v
	}
	sort.Float64s(out) // -100m … -0.02
	return out
}

// Signed funny: negatives then positives (−100m … −0.02, 0.02 … 100m).
func funnySignedLRs() []float64 {
	out := make([]float64, 0, len(defaultFunnyLRs)*2)
	out = append(out, funnyNegLRs()...)
	out = append(out, funnyLRs()...)
	return out
}

type Job struct {
	ID    string
	Kind  CellKind
	DType core.DType
	Mode  parallel.TrainMode
	LR    float64
}

type modeResult struct {
	ID      string        `json:"id"`
	Layer   string        `json:"layer"`
	DType   string        `json:"dtype"`
	Mode    string        `json:"mode"`
	LR      float64       `json:"lr"`
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
	layersFlag := flag.String("layers", EnvOr("TEST53_LAYERS", EnvOr("TEST53_LAYER", "dense")), "one layer or all|csv cell kinds")
	dtypesFlag := flag.String("dtypes", EnvOr("TEST53_DTYPES", "all"), "all|csv")
	lrsFlag := flag.String("lrs", EnvOr("TEST53_LRS", "funny-lo"),
		"funny-lo|lo = 0.02…2k; funny-hi|hi = 20k…100m; funny|all = full +ramp; funny-neg|neg; funny±|pm; or csv; empty = -lr once")
	camsFlag := flag.String("cams", EnvOr("TEST53_CAMS", "1"), "Welvet Parallel branch count (1=dense, 3=tricameral, …)")
	workersFlag := flag.Int("workers", EnvInt("TEST53_WORKERS", 0), "0 = NumCPU")
	dur := flag.Duration("duration", EnvDuration("TEST53_DURATION", 2*time.Second), "wall per job")
	lrOnce := flag.Float64("lr", EnvFloat("TEST53_LR", 0.05), "single LR when -lrs is empty")
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
	lrs, err := parseLRList(*lrsFlag, *lrOnce)
	must(err)
	nCams, err := parseCams(*camsFlag)
	must(err)

	ckptPath := resolveCkpt(*ckpt, kinds)

	workers := *workersFlag
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers < 1 {
		workers = 1
	}
	leafMultiCore = workers == 1

	jobs := expandJobs(kinds, modes, dtypes, lrs)
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  test53 · dayroute × layer × mode × dtype × funny-LR        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("task=dayroute  grid=%dx%d  days=%d  acts=%d  schedule=%v\n",
		gridSize, gridSize, nDays, nActs, []string{"wake", "bath", "breakfast", "work", "lunch", "gym", "couch", "sleep"})
	fmt.Printf("kinds=%d modes=%d dtypes=%d lrs=%d cams=%d jobs=%d workers=%d leafMultiCore=%v dur=%s\n",
		len(kinds), len(modes), len(dtypes), len(lrs), nCams, len(jobs), workers, leafMultiCore, *dur)
	fmt.Printf("kinds=%s\n", kindsCSV(kinds))
	fmt.Printf("lrs=%s  cams=%d (%s)\n", lrsCSV(lrs), nCams, camName(nCams))
	fmt.Printf("clock=%s  ckpt=%s\n\n", dutyClockName(), ckptPath)

	store := NewStore(ckptPath)
	prog, err := store.Load()
	must(err)
	done := doneSet(prog.DoneIDs)
	if !*resume {
		done = map[string]bool{}
		prog = &Progress{}
	}
	prog.Total = len(jobs)

	prior, _ := store.LoadResults()
	if *resume {
		mergeDoneIDs(prog, prior)
		done = doneSet(prog.DoneIDs)
	}
	results := make([]modeResult, 0, len(prior)+len(jobs))
	seenResult := map[string]bool{}
	for _, r := range prior {
		if r.ID == "" || seenResult[r.ID] {
			continue
		}
		seenResult[r.ID] = true
		results = append(results, r)
		done[r.ID] = true
	}
	if *resume && len(prior) > 0 {
		_ = store.SaveProgress(prog)
	}

	pending := make([]Job, 0, len(jobs))
	for _, j := range jobs {
		if !done[j.ID] {
			pending = append(pending, j)
		}
	}
	fmt.Printf("resume: done=%d pending=%d  ckpt=%d result(s) on disk\n",
		len(jobs)-len(pending), len(pending), len(results))

	tide := startTideBridge(*tideAddr, jobs, lrs[0], kinds, nCams)
	alreadyDone := len(jobs) - len(pending)
	if tide != nil {
		tide.seedCompleted(results)
		tide.resetPace() // ETA from new work only, not instant seed dumps
		tide.setQueue(alreadyDone, len(jobs), fmt.Sprintf("ready · %d/%d · %d left", alreadyDone, len(jobs), len(pending)))
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
		job     Job
		res     modeResult
		started time.Time
		ended   time.Time
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
				t0 := time.Now()
				res := runJob(j, seed, *hidden, nCams, *dur)
				outCh <- leafOut{job: j, res: res, started: t0, ended: time.Now()}
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
		// Replace prior row for this ID (crash/restart safe).
		replaced := false
		for i := range results {
			if results[i].ID == r.ID {
				results[i] = r
				replaced = true
				break
			}
		}
		if !replaced {
			results = append(results, r)
		}
		if tide != nil {
			// Commit with real wall times — do not re-Begin (that zeroed ETA).
			tide.commitJob(j, r, out.started, out.ended)
			left := planTotal - doneNow
			tide.setQueue(doneNow, planTotal, fmt.Sprintf("%s · %d/%d · %d left", r.ID, doneNow, planTotal, left))
		}
		if !done[r.ID] {
			prog.DoneIDs = append(prog.DoneIDs, r.ID)
		}
		done[r.ID] = true
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
			fmt.Printf("  [%d/%d pending · %d left] Acc %.1f Avail %.1f Score %.0f · %s · %s\n",
				n, totalPending, planTotal-doneNow, r.Acc, r.Avail, r.Score, r.ID, tag)
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

func expandJobs(kinds []CellKind, modes []parallel.TrainMode, dtypes []core.DType, lrs []float64) []Job {
	out := make([]Job, 0, len(kinds)*len(modes)*len(dtypes)*len(lrs))
	// LR↑ outermost: finish every mode × dtype × layer at one LR before the next.
	// Within an LR: mode → dtype → kind (kinds interleaved so Tide shows mha/lstm early).
	for _, lr := range lrs {
		for _, m := range modes {
			for _, dt := range dtypes {
				for _, k := range kinds {
					id := fmt.Sprintf("%s|%s|%s|lr=%s", k, dt, m.String(), formatLR(lr))
					out = append(out, Job{ID: id, Kind: k, DType: dt, Mode: m, LR: lr})
				}
			}
		}
	}
	return out
}

func parseLRList(spec string, once float64) ([]float64, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return []float64{once}, nil
	}
	switch strings.ToLower(spec) {
	case "funny-lo", "funny_lo", "lo", "mild", "low":
		return funnyLo(), nil
	case "funny-hi", "funny_hi", "hi", "high", "extreme":
		return funnyHi(), nil
	case "funny", "all", "pos", "+":
		return funnyLRs(), nil
	case "funny-neg", "funny_neg", "neg", "negative", "-":
		return funnyNegLRs(), nil
	case "funny±", "funny+/-", "funny+-", "pm", "signed", "±", "+/-", "+-":
		return funnySignedLRs(), nil
	}
	var out []float64
	seen := map[float64]bool{}
	for _, tok := range strings.FieldsFunc(spec, func(r rune) bool {
		return r == ',' || r == ' ' || r == ';'
	}) {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		v, err := parseLRToken(tok)
		if err != nil {
			return nil, err
		}
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no learning rates in %q", spec)
	}
	sort.Float64s(out)
	return out, nil
}

func parseLRToken(tok string) (float64, error) {
	low := strings.ToLower(strings.TrimSpace(tok))
	sign := 1.0
	if strings.HasPrefix(low, "-") {
		sign = -1
		low = strings.TrimPrefix(low, "-")
	} else if strings.HasPrefix(low, "+") {
		low = strings.TrimPrefix(low, "+")
	}
	mult := 1.0
	switch {
	case strings.HasSuffix(low, "m"):
		mult = 1e6
		low = strings.TrimSuffix(low, "m")
	case strings.HasSuffix(low, "k"):
		mult = 1e3
		low = strings.TrimSuffix(low, "k")
	}
	v, err := strconv.ParseFloat(low, 64)
	if err != nil {
		return 0, fmt.Errorf("lr %q: %w", tok, err)
	}
	return sign * v * mult, nil
}

func formatLR(lr float64) string {
	sign := ""
	x := lr
	if lr < 0 {
		sign = "-"
		x = -lr
	}
	switch {
	case x == 0:
		return "0"
	case x >= 1e6 && x == float64(int64(x/1e6))*1e6:
		return fmt.Sprintf("%s%dm", sign, int64(x/1e6))
	case x >= 1e3 && x == float64(int64(x/1e3))*1e3:
		return fmt.Sprintf("%s%dk", sign, int64(x/1e3))
	default:
		return sign + strconv.FormatFloat(x, 'g', -1, 64)
	}
}

func lrsCSV(lrs []float64) string {
	parts := make([]string, len(lrs))
	for i, lr := range lrs {
		parts[i] = formatLR(lr)
	}
	return strings.Join(parts, ",")
}

func parseModeList(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	var out []parallel.TrainMode
	if spec == "" || strings.EqualFold(spec, "all") {
		out = append(out, parallel.AllNamedTrainModes()...)
	} else {
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
	}
	out = filterRemovedModes(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("no modes in %q", spec)
	}
	return out, nil
}

func filterRemovedModes(in []parallel.TrainMode) []parallel.TrainMode {
	out := make([]parallel.TrainMode, 0, len(in))
	for _, m := range in {
		if removedTrainModes[m] {
			continue
		}
		out = append(out, m)
	}
	return out
}
func runJob(j Job, seed int64, hidden, nCams int, dur time.Duration) modeResult {
	lr := j.LR
	r := modeResult{
		ID: j.ID, Layer: string(j.Kind), DType: j.DType.String(), Mode: j.Mode.String(), LR: lr,
	}
	st, err := buildSandwich(j.Kind, obsDim, hidden, nActs, j.DType, nCams)
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
	r.Lucy.TotalOutputs = r.Actions
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
	fmt.Println("══ LPD top (layer × mode × dtype × lr) ══")
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
