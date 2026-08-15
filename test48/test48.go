package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/systems/dna"
)

// TEST 48 — test41 credit modes × every layer kind × xor/sine/copy × 1..3 cameral
// × FormatNone weight dtypes (activations stay float32; weight dtype ⊥ act T).
//
// Same Lucy measuring as test41_w_native_cam (SoftAcc, hard Acc, Avail, AdaptPct,
// Score, Tput, InferMs/TrainMs). Traditional backprop is StepBP (BackwardStack).
// Split-family jobs use OpenSplitTape (infer collect = train tape).
// Mesh modes are skipped (Dense-grid only). LinearCache is included but dead on sine.

const (
	defaultLR      = 0.05
	defaultHidden  = 32
	windowDuration = 50 * time.Millisecond
	sineWin        = 10
	sinePoints     = 64
	sineRes        = 0.1
)

type sample struct {
	x, y *core.Tensor[float32]
}

type toyTask struct {
	name     string
	in, out  int
	pools    [][]sample // sine: one pool per frequency; xor/copy: one pool
	eval     []sample
	binary   bool
	sineLike bool
}

type job struct {
	task  toyTask
	kind  CellKind
	nHemi int
	mode  parallel.TrainMode
	dt    core.DType
}

type row struct {
	Task    string        `json:"task"`
	DType   string        `json:"dtype"`
	Layer   string        `json:"layer"`
	Arch    string        `json:"arch"`
	Mode    string        `json:"mode"`
	Acc     float64       `json:"acc"`      // hard Acc (eval after the timed loop)
	Soft    float64       `json:"soft_acc"` // Lucy SoftAcc
	Adapt   float64       `json:"adapt_pct"`
	Avail   float64       `json:"availability"`
	Stab    float64       `json:"stability"`
	Cons    float64       `json:"consistency"`
	Tput    float64       `json:"throughput"`
	Score   float64       `json:"score"`
	Steps   int64         `json:"steps"`
	InferMs float64       `json:"infer_ms"`
	TrainMs float64       `json:"train_ms"`
	RAMKiB  float64       `json:"ram_kib"`
	Err     string        `json:"error,omitempty"`
	Lucy    lucy.Snapshot `json:"lucy"`
}

func main() {
	layersFlag := flag.String("layers", "all", "comma list, or all")
	camerals := flag.Int("camerals", 3, "max hemispheres (1..N, cap 3)")
	camMin := flag.Int("cam-min", 1, "first cameral count")
	only := flag.Int("only", 0, "exactly this many hemispheres")
	hidden := flag.Int("hidden", defaultHidden, "hidden width")
	dur := flag.Duration("duration", 2*time.Second, "wall per job (test41 long race: 10s)")
	switchEvery := flag.Duration("switch", 0, "sine freq switch (0 = duration/4)")
	adaptN := flag.Int("adapt-windows", 4, "pulse windows after switch folded into AdaptPct")
	lr := flag.Float64("lr", defaultLR, "learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	workers := flag.Int("workers", 0, "concurrent jobs (0 = NumCPU; 1 = Lucy-honest Score)")
	tasksFlag := flag.String("tasks", "xor,sine,copy", "xor,sine,copy")
	modesFlag := flag.String("modes", "all", "all = test41 stack modes, or comma list")
	dtypesFlag := flag.String("dtypes", "all", "all = 34 core.AllDTypes, or comma list (weight storage; act stays f32)")
	altTimes := flag.Int("alt-times", 1, "TweenAlt: Split→Tween pairs per sample")
	flag.Parse()

	kinds, err := parseLayerList("dense", *layersFlag, flag.Args())
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(2)
	}
	modes, err := parseModes(*modesFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(2)
	}
	dtypes, err := parseDTypeList(*dtypesFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(2)
	}
	if *only > 0 {
		*camMin, *camerals = *only, *only
	}
	if *camMin < 1 {
		*camMin = 1
	}
	if *camerals > 3 {
		*camerals = 3
	}
	if *camerals < *camMin {
		*camerals = *camMin
	}
	if *hidden < 8 {
		*hidden = 8
	}
	if *altTimes < 1 {
		*altTimes = 1
	}
	if *workers <= 0 {
		*workers = runtime.NumCPU()
	}
	if *adaptN <= 0 {
		*adaptN = lucy.AdaptWindowsDefault
	}
	sw := *switchEvery
	if sw <= 0 {
		sw = *dur / 4
		if sw < 50*time.Millisecond {
			sw = 50 * time.Millisecond
		}
	}
	rand.Seed(*seed)

	tasks := buildTasks(*tasksFlag, *seed)
	if len(tasks) == 0 {
		fmt.Fprintf(os.Stderr, "❌ no tasks\n")
		os.Exit(2)
	}

	var jobs []job
	for _, t := range tasks {
		for _, dt := range dtypes {
			for _, k := range kinds {
				for n := *camMin; n <= *camerals; n++ {
					for _, m := range modes {
						jobs = append(jobs, job{task: t, kind: k, nHemi: n, mode: m, dt: dt})
					}
				}
			}
		}
	}

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   TEST 48 — test41 modes × all layers × xor/sine/copy × Dense..Tricameral × dtypes   ║")
	fmt.Println("║   StepBP = backprop. Lucy: Acc / SoftAcc / Avail / AdaptPct / Score                 ║")
	fmt.Println("║   Weight dtype = FormatNone storage; activations stay float32 (cross-numeric).      ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("🧠 layers=%s\n", kindsCSV(kinds))
	fmt.Printf("🔢 dtypes=%s (%d)\n", dtypesCSV(dtypes), len(dtypes))
	fmt.Printf("📐 cams=%d..%d  hidden=%d  duration=%s/job  switch=%s  lr=%.3f  alt-times=%d\n",
		*camMin, *camerals, *hidden, *dur, sw, *lr, *altTimes)
	fmt.Printf("🧪 tasks=%s  modes=%s\n", *tasksFlag, modesCSV(modes))
	fmt.Printf("📊 jobs=%d  workers=%d  duty=%s  adapt-windows=%d\n\n",
		len(jobs), *workers, dutyClockName(), *adaptN)

	rows := make([]row, len(jobs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)
	for i, j := range jobs {
		wg.Add(1)
		go func(i int, j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			defer func() {
				if rec := recover(); rec != nil {
					rows[i] = row{
						Task:  j.task.name,
						DType: j.dt.String(),
						Layer: string(j.kind),
						Arch:  camName(j.nHemi),
						Mode:  j.mode.String(),
						Err:   fmt.Sprintf("panic: %v", rec),
					}
					fmt.Printf("❌ [%s %s %s/%s %s] panic: %v\n",
						j.task.name, j.dt, j.kind, camName(j.nHemi), j.mode, rec)
				}
			}()
			rows[i] = runLucyJob(j, *hidden, *dur, sw, *lr, *altTimes, *adaptN)
			r := rows[i]
			tag := fmt.Sprintf("%s %s %s/%s %s", r.Task, r.DType, r.Layer, r.Arch, r.Mode)
			if r.Err != "" {
				fmt.Printf("❌ [%s] %s\n", tag, r.Err)
				return
			}
			fmt.Printf("✅ [%s] Acc:%.1f%% Soft:%.1f%% Avail:%.1f%% Adapt:%.1f%% Tput:%.0f Score:%.0f\n",
				tag, r.Acc, r.Soft, r.Avail, r.Adapt, r.Tput, r.Score)
		}(i, j)
	}
	wg.Wait()

	data, _ := json.MarshalIndent(map[string]any{
		"engine":        "welvet-test48",
		"duration":      dur.String(),
		"switch":        sw.String(),
		"alt_times":     *altTimes,
		"workers":       *workers,
		"adapt_windows": *adaptN,
		"dtypes":        dtypesCSV(dtypes),
		"score_formula": "Throughput × Availability × SoftAcc / 10000",
		"rows":          rows,
	}, "", "  ")
	_ = os.WriteFile("test48_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test48_results.json")
	printSummary(rows, tasks, dtypes)
}

func allTest41StackModes() []parallel.TrainMode {
	return []parallel.TrainMode{
		parallel.ModeNormalBP, parallel.ModeStepBP,
		parallel.ModeTween, parallel.ModeTweenChain,
		parallel.ModeStepTween, parallel.ModeStepTweenChain,
		parallel.ModeTweenSplit, parallel.ModeStepTweenSplit,
		parallel.ModeTweenAlt, parallel.ModeStepTweenAlt,
		parallel.ModeTweenSplitHeadProxy, parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse,
	}
}

func parseModes(s string) ([]parallel.TrainMode, error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "all") {
		return allTest41StackModes(), nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, p := range splitLayerTokens(s) {
		var m parallel.TrainMode
		switch strings.ToLower(p) {
		case "normalbp", "bp", "sgd":
			m = parallel.ModeNormalBP
		case "stepbp":
			m = parallel.ModeStepBP
		case "tween":
			m = parallel.ModeTween
		case "tweenchain", "chain":
			m = parallel.ModeTweenChain
		case "steptween":
			m = parallel.ModeStepTween
		case "steptweenchain":
			m = parallel.ModeStepTweenChain
		case "tweensplit", "split":
			m = parallel.ModeTweenSplit
		case "steptweensplit", "stepsplit":
			m = parallel.ModeStepTweenSplit
		case "tweenalt", "alt":
			m = parallel.ModeTweenAlt
		case "steptweenalt", "stepalt":
			m = parallel.ModeStepTweenAlt
		case "headproxy", "tweensplitheadproxy", "proxy":
			m = parallel.ModeTweenSplitHeadProxy
		case "linear", "tweensplitlinear":
			m = parallel.ModeTweenSplitLinear
		case "fastproxy", "tweensplitfastproxy":
			m = parallel.ModeTweenSplitFastProxy
		case "linearcache", "tweensplitlinearcache", "cachedlinear":
			m = parallel.ModeTweenSplitLinearCache
		case "proxyasync", "headproxyasync", "async", "tweensplitheadproxyasync":
			m = parallel.ModeTweenSplitHeadProxyAsync
		case "sparse", "tweensplitsparse":
			m = parallel.ModeTweenSplitSparse
		default:
			return nil, fmt.Errorf("unknown mode %q", p)
		}
		if seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no modes")
	}
	return out, nil
}

func modesCSV(ms []parallel.TrainMode) string {
	parts := make([]string, len(ms))
	for i, m := range ms {
		parts[i] = m.String()
	}
	return strings.Join(parts, ",")
}

func buildTasks(spec string, seed int64) []toyTask {
	var out []toyTask
	for _, p := range splitLayerTokens(spec) {
		switch strings.ToLower(p) {
		case "xor":
			out = append(out, makeXOR())
		case "sine":
			out = append(out, makeSineAdapt())
		case "copy":
			out = append(out, makeCopy(seed, 16, 24, 8))
		}
	}
	return out
}

func vec(n int, vals ...float32) *core.Tensor[float32] {
	t := core.NewTensor[float32](1, n)
	copy(t.Data, vals)
	return t
}

func makeXOR() toyTask {
	const dim = 16
	mk := func(a, b, y float32) sample {
		x := vec(dim)
		x.Data[0], x.Data[1] = a, b
		return sample{x: x, y: vec(1, y)}
	}
	set := []sample{
		mk(0, 0, 0), mk(0, 1, 1), mk(1, 0, 1), mk(1, 1, 0),
	}
	return toyTask{name: "xor", in: dim, out: 1, pools: [][]sample{set}, eval: set, binary: true}
}

func makeSineAdapt() toyTask {
	const dim = 16
	freqs := []float64{1, 2, 3, 4}
	pools := make([][]sample, len(freqs))
	for i, f := range freqs {
		pools[i] = sinePool(dim, f)
	}
	return toyTask{
		name: "sine", in: dim, out: 1, pools: pools, eval: pools[0],
		binary: false, sineLike: true,
	}
}

func sinePool(dim int, freq float64) []sample {
	win := sineWin
	if win > dim {
		win = dim
	}
	n := sinePoints - win
	if n < 8 {
		n = 8
	}
	out := make([]sample, n)
	wave := make([]float64, sinePoints)
	for i := range wave {
		wave[i] = math.Sin(freq * float64(i) * sineRes)
	}
	for i := 0; i < n; i++ {
		x := vec(dim)
		for j := 0; j < win; j++ {
			x.Data[j] = float32((wave[i+j] + 1) / 2)
		}
		y := float32((wave[i+win] + 1) / 2)
		out[i] = sample{x: x, y: vec(1, y)}
	}
	return out
}

func makeCopy(seed int64, dim, nTrain, nEval int) toyTask {
	rng := rand.New(rand.NewSource(seed + 7))
	mk := func() sample {
		x := vec(dim)
		y := vec(dim)
		for i := 0; i < dim; i++ {
			if rng.Float32() < 0.5 {
				x.Data[i] = 1
			}
			y.Data[i] = x.Data[i]
		}
		return sample{x: x, y: y}
	}
	train := make([]sample, nTrain)
	eval := make([]sample, nEval)
	for i := range train {
		train[i] = mk()
	}
	for i := range eval {
		eval[i] = mk()
	}
	return toyTask{name: "copy", in: dim, out: dim, pools: [][]sample{train}, eval: eval, binary: true}
}

func runLucyJob(j job, hidden int, dur, switchEvery time.Duration, lr float64, altTimes, adaptN int) row {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r := row{
		Task:  j.task.name,
		DType: j.dt.String(),
		Layer: string(j.kind),
		Arch:  camName(j.nHemi),
		Mode:  j.mode.String(),
	}
	stack, err := buildNativeCameral(j.kind, j.task.in, hidden, j.task.out, j.nHemi, j.mode, j.dt)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	stack.AltTimes = altTimes
	r.Lucy.WeightBytes = stackWeightBytes(stack)

	numWindows := int(dur / windowDuration)
	if numWindows < 1 {
		numWindows = 1
	}
	win := make([]lucy.Window, numWindows)

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	poolIdx := 0
	lastSwitch := start
	var totalInfer, totalTrain time.Duration

	for time.Since(start) < dur {
		elapsed := time.Since(start)
		newWindow := int(elapsed / windowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}
		if j.task.sineLike && len(j.task.pools) > 1 &&
			time.Since(lastSwitch) >= switchEvery && poolIdx < len(j.task.pools)-1 {
			poolIdx++
			lastSwitch = time.Now()
			if currentWindow < numWindows {
				win[currentWindow].PhaseSwitches++
			}
		}
		pool := j.task.pools[poolIdx]
		s := pool[sampleIdx%len(pool)]
		sampleIdx++

		tInf := startWork()
		var post *core.Tensor[float32]
		var tape *parallel.SplitTape[float32]
		if j.mode.IsSplitFamily() {
			tp, terr := parallel.OpenSplitTape(stack, s.x)
			if terr != nil || tp == nil || tp.Post == nil {
				continue
			}
			tape = tp
			post = tp.Post
		} else {
			_, p, ferr := parallel.ForwardStack(stack, s.x)
			if ferr != nil || p == nil {
				continue
			}
			post = p
		}
		inferDur := tInf.elapsed()
		totalInfer += inferDur

		hard, soft := scoreSample(post, s.y, j.task)
		r.Lucy.TotalOutputs++
		r.Steps++
		if currentWindow < numWindows {
			w := &win[currentWindow]
			w.Outputs++
			w.InferMs += inferDur.Seconds() * 1000
			w.SoftAcc += soft
			w.Accuracy += hard
		}

		t0 := startWork()
		if tape != nil {
			_, _ = tape.Train(s.y, j.mode, lr)
		} else {
			_, _ = parallel.TrainStackMSE(stack, s.x, s.y, j.mode, lr)
		}
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
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: adaptN, ConsThreshold: lucy.ConsThreshold})

	evalPool := j.task.eval
	if j.task.sineLike && poolIdx < len(j.task.pools) {
		evalPool = j.task.pools[poolIdx]
	}
	r.Acc, _ = scoreTask(stack, j.task, evalPool)
	r.Soft = r.Lucy.SoftAcc
	r.Adapt = r.Lucy.AdaptPct
	r.Avail = r.Lucy.Availability
	r.Stab = r.Lucy.Stability
	r.Cons = r.Lucy.Consistency
	r.Tput = r.Lucy.Throughput
	r.Score = r.Lucy.Score
	r.InferMs = r.Lucy.InferMs
	r.TrainMs = r.Lucy.TrainMs
	r.RAMKiB = float64(r.Lucy.WeightBytes) / 1024
	return r
}

func scoreSample(post, target *core.Tensor[float32], task toyTask) (hard, soft float64) {
	if post == nil || target == nil {
		return 0, 0
	}
	n := post.Len()
	if target.Len() < n {
		n = target.Len()
	}
	if n == 0 {
		return 0, 0
	}
	if task.binary {
		ok := 0
		var s float64
		for i := 0; i < n; i++ {
			pred, gold := 0.0, 0.0
			if post.Data[i] >= 0.5 {
				pred = 1
			}
			if target.Data[i] >= 0.5 {
				gold = 1
			}
			if pred == gold {
				ok++
			}
			s += lucy.SoftAccProb(post.Data[i], target.Data[i])
		}
		return 100 * float64(ok) / float64(n), s / float64(n)
	}
	all := true
	var s float64
	for i := 0; i < n; i++ {
		p := float64(post.Data[i])
		t := float64(target.Data[i])
		if math.Abs(p-t) > 0.15 {
			all = false
		}
		s += lucy.SoftAccOne(post.Data[i], target.Data[i])
	}
	if all {
		hard = 100
	}
	return hard, s / float64(n)
}

func scoreTask(stack *parallel.Stack, task toyTask, eval []sample) (acc, soft float64) {
	if len(eval) == 0 {
		return 0, 0
	}
	var hardSum, softSum float64
	n := 0
	for _, s := range eval {
		_, post, err := parallel.ForwardStack(stack, s.x)
		if err != nil || post == nil {
			continue
		}
		h, so := scoreSample(post, s.y, task)
		hardSum += h
		softSum += so
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return hardSum / float64(n), softSum / float64(n)
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

func printSummary(rows []row, tasks []toyTask, dtypes []core.DType) {
	byTask := map[string][]row{}
	for _, r := range rows {
		byTask[r.Task] = append(byTask[r.Task], r)
	}
	for _, t := range tasks {
		all := byTask[t.name]
		dts := dtypes
		if len(dts) == 0 {
			dts = dtypesIn(all)
		}
		for _, dt := range dts {
			var rs []row
			for _, r := range all {
				if r.DType == dt.String() {
					rs = append(rs, r)
				}
			}
			if len(rs) == 0 {
				continue
			}
			fmt.Println()
			fmt.Printf("══ %s  dtype=%s  in=%d out=%d  (Acc = hard eval; Soft/Avail/Score = Lucy live) ══\n",
				t.name, dt, t.in, t.out)
			fmt.Printf("%-52s │ %6s %6s %6s %6s %7s %8s\n",
				"layer/arch/mode", "Acc", "Soft", "Avail", "Adapt", "Tput", "Score")
			sort.Slice(rs, func(i, j int) bool {
				if rs[i].Score != rs[j].Score {
					return rs[i].Score > rs[j].Score
				}
				return rs[i].Layer+rs[i].Arch+rs[i].Mode < rs[j].Layer+rs[j].Arch+rs[j].Mode
			})
			best := -1.0
			bestName := ""
			for _, r := range rs {
				name := fmt.Sprintf("%s/%s/%s", r.Layer, r.Arch, r.Mode)
				if r.Err != "" {
					fmt.Printf("%-52s │ ERR %s\n", name, trimErr(r.Err, 40))
					continue
				}
				fmt.Printf("%-52s │ %5.1f%% %5.1f%% %5.1f%% %5.1f%% %7.0f %8.0f\n",
					name, r.Acc, r.Soft, r.Avail, r.Adapt, r.Tput, r.Score)
				if r.Score > best {
					best, bestName = r.Score, name
				}
			}
			fmt.Printf("   🏆 Score winner: %s/%s  %.0f\n", dt, bestName, best)
			printVsBP(rs)
		}
	}
}

func dtypesIn(rs []row) []core.DType {
	seen := map[string]bool{}
	var out []core.DType
	for _, dt := range core.AllDTypes {
		for _, r := range rs {
			if r.DType == dt.String() && !seen[r.DType] {
				seen[r.DType] = true
				out = append(out, dt)
				break
			}
		}
	}
	return out
}

func printVsBP(rs []row) {
	type key struct{ layer, arch string }
	bp := map[key]row{}
	for _, r := range rs {
		if r.Mode == parallel.ModeStepBP.String() || r.Mode == parallel.ModeNormalBP.String() {
			k := key{r.Layer, r.Arch}
			if _, ok := bp[k]; !ok || r.Mode == parallel.ModeStepBP.String() {
				bp[k] = r
			}
		}
	}
	if len(bp) == 0 {
		return
	}
	fmt.Println("   vs StepBP (Score Δ)  FastProxy / Sparse / HeadProxy / Linear:")
	var keys []key
	for k := range bp {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return keys[i].arch < keys[j].arch
	})
	by := map[key]map[string]row{}
	for _, r := range rs {
		k := key{r.Layer, r.Arch}
		if by[k] == nil {
			by[k] = map[string]row{}
		}
		by[k][r.Mode] = r
	}
	want := []string{
		parallel.ModeTweenSplitFastProxy.String(),
		parallel.ModeTweenSplitSparse.String(),
		parallel.ModeTweenSplitHeadProxy.String(),
		parallel.ModeTweenSplitLinear.String(),
	}
	for _, k := range keys {
		b := bp[k]
		if b.Err != "" {
			continue
		}
		fmt.Printf("     %-12s %-12s  BP Acc %.0f Score %.0f │", k.layer, k.arch, b.Acc, b.Score)
		for _, m := range want {
			mar := by[k][m]
			if mar.Err != "" || mar.Mode == "" {
				fmt.Printf("  %s —", shortMode(m))
				continue
			}
			fmt.Printf("  %s Acc%+.0f Score%+.0f", shortMode(m), mar.Acc-b.Acc, mar.Score-b.Score)
		}
		fmt.Println()
	}
}

func shortMode(m string) string {
	switch m {
	case "TweenSplitFastProxy":
		return "FastProxy"
	case "TweenSplitSparse":
		return "Sparse"
	case "TweenSplitHeadProxy":
		return "HeadProxy"
	case "TweenSplitLinear":
		return "Linear"
	default:
		return m
	}
}

func trimErr(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n]
	}
	return s
}
