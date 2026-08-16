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

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/dna"
)

// TEST 50 — deep FP32 Lucy race: every named train mode (stack + Mesh*) on
// xor / sine / copy × 1³/2³/3³. Same Lucy measuring as test41 / test48.
// One live sandwich at the origin (rest disabled) — permutation + long train,
// not 27 copies. MeshBP = volumetric Step; MeshTween = StepMesh; MeshTweenChain
// = StepTween; Mesh Split/Alt/FastProxy/Sparse credit on the placed stack.

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
	pools    [][]sample
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
	gridN int
}

type row struct {
	Task    string        `json:"task"`
	DType   string        `json:"dtype"`
	Layer   string        `json:"layer"`
	Arch    string        `json:"arch"`
	Grid    string        `json:"grid"`
	Mode    string        `json:"mode"`
	Acc     float64       `json:"acc"`
	Soft    float64       `json:"soft_acc"`
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

type winner struct {
	Task  string  `json:"task"`
	Layer string  `json:"layer"`
	Arch  string  `json:"arch"`
	Grid  string  `json:"grid"`
	Axis  string  `json:"axis"`
	Mode  string  `json:"mode"`
	Acc   float64 `json:"acc"`
	Score float64 `json:"score"`
	VsBP  string  `json:"vs_stepbp,omitempty"`
}

func main() {
	layersFlag := flag.String("layers", "dense", "comma list, or all (default dense — the comparable Lucy sandwich)")
	camerals := flag.Int("camerals", 3, "max hemispheres (1..N, cap 3)")
	camMin := flag.Int("cam-min", 1, "first cameral count")
	only := flag.Int("only", 0, "exactly this many hemispheres")
	hidden := flag.Int("hidden", defaultHidden, "hidden width")
	dur := flag.Duration("duration", time.Second, "wall per job")
	switchEvery := flag.Duration("switch", 0, "sine freq switch (0 = duration/4)")
	adaptN := flag.Int("adapt-windows", 10, "pulse windows after switch folded into AdaptPct")
	lr := flag.Float64("lr", defaultLR, "learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	workers := flag.Int("workers", 1, "concurrent jobs (1 = Lucy-honest Score/Avail)")
	tasksFlag := flag.String("tasks", "xor,sine,copy", "xor,sine,copy")
	modesFlag := flag.String("modes", "all", "all = every named mode (23, incl. Mesh*), or comma list")
	gridsFlag := flag.String("grids", "1,2,3", "cube sizes: 1,2,3 or 1x1x1,2x2x2")
	dtypesFlag := flag.String("dtypes", "float32", "weight storage; act stays f32")
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
	grids, err := parseGrids(*gridsFlag)
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
		*workers = 1
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
					for _, gn := range grids {
						for _, m := range modes {
							jobs = append(jobs, job{task: t, kind: k, nHemi: n, mode: m, dt: dt, gridN: gn})
						}
					}
				}
			}
		}
	}

	eta := time.Duration(int64(math.Ceil(float64(len(jobs))*dur.Seconds()/float64(*workers)))) * time.Second

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   TEST 50 — deep FP32 Lucy race · ALL named modes × 1³/2³/3³ origin smoke           ║")
	fmt.Println("║   StepBP = backprop rival (hard Acc). Score = Tput × Avail × SoftAcc / 10000        ║")
	fmt.Println("║   One live cell at origin. LinearCache is the sine-dead control.                    ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("🧠 layers=%s\n", kindsCSV(kinds))
	fmt.Printf("🔢 dtypes=%s (%d)\n", dtypesCSV(dtypes), len(dtypes))
	fmt.Printf("📦 grids=%s  (origin-only, rest disabled)\n", gridsCSV(grids))
	fmt.Printf("📐 cams=%d..%d  hidden=%d  duration=%s/job  switch=%s  lr=%.3f  alt-times=%d\n",
		*camMin, *camerals, *hidden, *dur, sw, *lr, *altTimes)
	fmt.Printf("🧪 tasks=%s  modes=%s (%d)\n", *tasksFlag, modesCSV(modes), len(modes))
	fmt.Printf("📊 jobs=%d  workers=%d  duty=%s  adapt-windows=%d  ETA≈%s\n\n",
		len(jobs), *workers, dutyClockName(), *adaptN, eta)

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
						Grid:  gridName(j.gridN),
						Mode:  j.mode.String(),
						Err:   fmt.Sprintf("panic: %v", rec),
					}
					fmt.Printf("❌ [%s %s %s/%s/%s %s] panic: %v\n",
						j.task.name, j.dt, j.kind, camName(j.nHemi), gridName(j.gridN), j.mode, rec)
				}
			}()
			rows[i] = runLucyJob(j, *hidden, *dur, sw, *lr, *altTimes, *adaptN)
			r := rows[i]
			tag := fmt.Sprintf("%s %s %s/%s/%s %s", r.Task, r.DType, r.Layer, r.Arch, r.Grid, r.Mode)
			if r.Err != "" {
				fmt.Printf("❌ [%s] %s\n", tag, r.Err)
				return
			}
			fmt.Printf("✅ [%s] Acc:%.1f%% Soft:%.1f%% Avail:%.1f%% Adapt:%.1f%% Tput:%.0f Score:%.0f steps:%d\n",
				tag, r.Acc, r.Soft, r.Avail, r.Adapt, r.Tput, r.Score, r.Steps)
		}(i, j)
	}
	wg.Wait()

	winners := collectWinners(rows)
	data, _ := json.MarshalIndent(map[string]any{
		"engine":        "welvet-test50",
		"duration":      dur.String(),
		"switch":        sw.String(),
		"alt_times":     *altTimes,
		"workers":       *workers,
		"adapt_windows": *adaptN,
		"dtypes":        dtypesCSV(dtypes),
		"grids":         gridsCSV(grids),
		"score_formula": "Throughput × Availability × SoftAcc / 10000",
		"rival":         "hard Acc vs StepBP — not Lucy Score",
		"rows":          rows,
		"winners":       winners,
	}, "", "  ")
	_ = os.WriteFile("test50_results.json", data, 0644)
	slim, _ := json.MarshalIndent(winners, "", "  ")
	_ = os.WriteFile("test50_winners.json", slim, 0644)
	fmt.Println("\n✅ Results saved to test50_results.json")
	fmt.Println("✅ Winners saved to test50_winners.json")
	printSummary(rows, tasks, dtypes)
	printLeaderboards(rows)
}

func parseModes(s string) ([]parallel.TrainMode, error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "all") {
		return parallel.AllNamedTrainModes(), nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, p := range splitLayerTokens(s) {
		m, err := parallel.ParseTrainMode(p)
		if err != nil {
			return nil, fmt.Errorf("unknown mode %q", p)
		}
		if m == parallel.ModeInherit {
			return nil, fmt.Errorf("mode %q is Inherit (not an update)", p)
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

func parseGrids(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "all") {
		return []int{1, 2, 3}, nil
	}
	var out []int
	seen := map[int]bool{}
	for _, p := range splitLayerTokens(s) {
		p = strings.ToLower(strings.TrimSpace(p))
		p = strings.ReplaceAll(p, "×", "x")
		n := 0
		switch {
		case p == "1" || p == "1x1x1":
			n = 1
		case p == "2" || p == "2x2x2":
			n = 2
		case p == "3" || p == "3x3x3":
			n = 3
		default:
			return nil, fmt.Errorf("unknown grid %q (want 1,2,3)", p)
		}
		if seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no grids")
	}
	sort.Ints(out)
	return out, nil
}

func gridName(n int) string {
	if n < 1 {
		n = 1
	}
	return fmt.Sprintf("%dx%dx%d", n, n, n)
}

func gridsCSV(ns []int) string {
	parts := make([]string, len(ns))
	for i, n := range ns {
		parts[i] = gridName(n)
	}
	return strings.Join(parts, ",")
}

func originGrid(n int) *architecture.Grid {
	if n < 1 {
		n = 1
	}
	g := architecture.NewGrid(n, n, n, 1)
	for i := range g.Cells {
		g.Cells[i].Layer.IsDisabled = true
	}
	return g
}

func enableOrigin(g *architecture.Grid) {
	if c := g.At(0, 0, 0, 0); c != nil {
		c.Layer.IsDisabled = false
	}
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
		Grid:  gridName(j.gridN),
		Mode:  j.mode.String(),
	}
	stack, err := buildNativeCameral(j.kind, j.task.in, hidden, j.task.out, j.nHemi, j.mode, j.dt)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	stack.AltTimes = altTimes
	r.Lucy.WeightBytes = stackWeightBytes(stack)

	grid := originGrid(j.gridN)
	if err := parallel.PlaceStack(grid, 0, 0, 0, 0, stack); err != nil {
		r.Err = "place: " + err.Error()
		return r
	}
	enableOrigin(grid)

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
		var meshFwd *forward.Result[float32]
		if j.mode == parallel.ModeMeshBP || j.mode == parallel.ModeMeshTween || j.mode == parallel.ModeMeshTweenChain {
			fwd, ferr := forward.Forward(grid, s.x)
			if ferr != nil || fwd == nil || fwd.Output == nil {
				continue
			}
			meshFwd = fwd
			post = fwd.Output
		} else if j.mode.IsSplitFamily() {
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
		_ = trainLucySample(stack, grid, meshFwd, tape, s.x, s.y, j.mode, lr)
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

func trainLucySample(stack *parallel.Stack, grid *architecture.Grid, meshFwd *forward.Result[float32], tape *parallel.SplitTape[float32], x, y *core.Tensor[float32], mode parallel.TrainMode, lr float64) error {
	switch mode {
	case parallel.ModeMeshBP:
		fwd := meshFwd
		if fwd == nil {
			var err error
			fwd, err = forward.Forward(grid, x)
			if err != nil {
				return err
			}
		}
		_, err := training.Step(fwd, y, lr)
		return err
	case parallel.ModeMeshTween:
		_, _, err := training.StepMesh(grid, x, y, 1, lr)
		return err
	case parallel.ModeMeshTweenChain:
		_, _, err := training.StepTween(grid, x, y, lr)
		return err
	}
	if tape != nil {
		_, err := tape.Train(y, mode, lr)
		return err
	}
	_, err := parallel.TrainStackMSE(stack, x, y, mode, lr)
	return err
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
			fmt.Printf("%-60s │ %6s %6s %6s %6s %7s %8s\n",
				"layer/arch/grid/mode", "Acc", "Soft", "Avail", "Adapt", "Tput", "Score")
			sort.Slice(rs, func(i, j int) bool {
				if rs[i].Acc != rs[j].Acc {
					return rs[i].Acc > rs[j].Acc
				}
				if rs[i].Score != rs[j].Score {
					return rs[i].Score > rs[j].Score
				}
				return rs[i].Layer+rs[i].Arch+rs[i].Grid+rs[i].Mode < rs[j].Layer+rs[j].Arch+rs[j].Grid+rs[j].Mode
			})
			for _, r := range rs {
				name := fmt.Sprintf("%s/%s/%s/%s", r.Layer, r.Arch, r.Grid, r.Mode)
				if r.Err != "" {
					fmt.Printf("%-60s │ ERR %s\n", name, trimErr(r.Err, 40))
					continue
				}
				fmt.Printf("%-60s │ %5.1f%% %5.1f%% %5.1f%% %5.1f%% %7.0f %8.0f\n",
					name, r.Acc, r.Soft, r.Avail, r.Adapt, r.Tput, r.Score)
			}
			printVsBP(rs)
		}
	}
}

func boardKey(r row) string {
	return r.Task + "|" + r.DType + "|" + r.Layer + "|" + r.Arch + "|" + r.Grid
}

func okRows(rs []row) []row {
	var out []row
	for _, r := range rs {
		if r.Err == "" {
			out = append(out, r)
		}
	}
	return out
}

func collectWinners(rows []row) []winner {
	groups := map[string][]row{}
	for _, r := range okRows(rows) {
		k := boardKey(r)
		groups[k] = append(groups[k], r)
	}
	var keys []string
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []winner
	for _, k := range keys {
		rs := groups[k]
		if len(rs) == 0 {
			continue
		}
		bp := stepBP(rs)
		add := func(axis string, pick row) {
			w := winner{
				Task: pick.Task, Layer: pick.Layer, Arch: pick.Arch, Grid: pick.Grid,
				Axis: axis, Mode: pick.Mode, Acc: pick.Acc, Score: pick.Score,
			}
			if bp != nil && pick.Mode != bp.Mode {
				w.VsBP = fmt.Sprintf("Acc%+.1f Score%+.0f", pick.Acc-bp.Acc, pick.Score-bp.Score)
			}
			out = append(out, w)
		}
		acc := append([]row(nil), rs...)
		sort.Slice(acc, func(i, j int) bool {
			if acc[i].Acc != acc[j].Acc {
				return acc[i].Acc > acc[j].Acc
			}
			return acc[i].Mode < acc[j].Mode
		})
		score := append([]row(nil), rs...)
		sort.Slice(score, func(i, j int) bool {
			if score[i].Score != score[j].Score {
				return score[i].Score > score[j].Score
			}
			return score[i].Mode < score[j].Mode
		})
		add("hard Acc", acc[0])
		add("Lucy Score", score[0])
	}
	return out
}

func stepBP(rs []row) *row {
	var bp *row
	for i := range rs {
		if rs[i].Mode == parallel.ModeStepBP.String() {
			bp = &rs[i]
			break
		}
		if rs[i].Mode == parallel.ModeNormalBP.String() && bp == nil {
			bp = &rs[i]
		}
	}
	return bp
}

func printLeaderboards(rows []row) {
	groups := map[string][]row{}
	for _, r := range okRows(rows) {
		k := boardKey(r)
		groups[k] = append(groups[k], r)
	}
	var keys []string
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Println()
	fmt.Println("══ TEST50 LEADERBOARDS — who won the deep run ══")
	fmt.Println("   Rival = hard Acc vs StepBP. Lucy Score rewards Tput/Avail; Sparse can win Score and lose Acc.")
	for _, k := range keys {
		rs := groups[k]
		r0 := rs[0]
		fmt.Printf("\n   %s / %s / %s / %s / %s\n", r0.Task, r0.DType, r0.Layer, r0.Arch, r0.Grid)
		bp := stepBP(rs)
		acc := append([]row(nil), rs...)
		sort.Slice(acc, func(i, j int) bool {
			if acc[i].Acc != acc[j].Acc {
				return acc[i].Acc > acc[j].Acc
			}
			if acc[i].Score != acc[j].Score {
				return acc[i].Score > acc[j].Score
			}
			return acc[i].Mode < acc[j].Mode
		})
		fmt.Printf("     Acc rank:\n")
		for i, r := range acc {
			delta := ""
			if bp != nil && r.Mode != bp.Mode {
				delta = fmt.Sprintf("  AccΔ%+.1f", r.Acc-bp.Acc)
			}
			mark := "  "
			if i == 0 {
				mark = "🏆"
			}
			fmt.Printf("     %s %2d. %-28s Acc %5.1f%%  Score %8.0f  Soft %5.1f%%%s\n",
				mark, i+1, r.Mode, r.Acc, r.Score, r.Soft, delta)
		}
		score := append([]row(nil), rs...)
		sort.Slice(score, func(i, j int) bool {
			if score[i].Score != score[j].Score {
				return score[i].Score > score[j].Score
			}
			return score[i].Mode < score[j].Mode
		})
		fmt.Printf("     🏅 Score winner: %s  (%.0f)   🏆 Acc winner: %s  (%.1f%%)\n",
			score[0].Mode, score[0].Score, acc[0].Mode, acc[0].Acc)
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
	type key struct{ layer, arch, grid string }
	bp := map[key]row{}
	for _, r := range rs {
		if r.Mode == parallel.ModeStepBP.String() || r.Mode == parallel.ModeNormalBP.String() {
			k := key{r.Layer, r.Arch, r.Grid}
			if _, ok := bp[k]; !ok || r.Mode == parallel.ModeStepBP.String() {
				bp[k] = r
			}
		}
	}
	if len(bp) == 0 {
		return
	}
	fmt.Println("   vs StepBP (Acc Δ / Score Δ)  FastProxy / Sparse / HeadProxy / Linear:")
	var keys []key
	for k := range bp {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		if keys[i].arch != keys[j].arch {
			return keys[i].arch < keys[j].arch
		}
		return keys[i].grid < keys[j].grid
	})
	by := map[key]map[string]row{}
	for _, r := range rs {
		k := key{r.Layer, r.Arch, r.Grid}
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
		fmt.Printf("     %-12s %-12s %-8s  BP Acc %.0f Score %.0f │", k.layer, k.arch, k.grid, b.Acc, b.Score)
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
