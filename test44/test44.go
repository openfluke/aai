package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/weights"
)

// TEST 44 — ARC-AGI native camerals (welvet v0.95.1)
//
// One Sandwich net per arch (Dense + Hemispheres 2..camerals). Cycle every
// training-split demo pair for -item-time (default 125ms) of StepTweenChain,
// pulse-infer after each item (Lucy/tide metrics), then score training-task
// test grids and the evaluation split. Dense now; -layer reserved.

const defaultLR = 0.01

type job struct {
	nHemi int
}

type SplitScore struct {
	Name       string   `json:"name"`
	Tasks      int      `json:"tasks"`
	TestGrids  int      `json:"test_grids"`
	ExactGrids int      `json:"exact_grids"`
	Solved     int      `json:"solved"`
	MeanPixel  float64  `json:"mean_pixel"`
	MeanSoft   float64  `json:"mean_soft"`
	SolvedIDs  []string `json:"solved_ids,omitempty"`
}

type ArchResult struct {
	Label      string        `json:"label"`
	Arch       string        `json:"arch"`
	Hemis      int           `json:"hemispheres"`
	Layer      string        `json:"layer"`
	Mode       string        `json:"mode"`
	Items      int           `json:"train_items"`
	Passes     int           `json:"passes"`
	ItemTime   string        `json:"item_time"`
	TrainSteps int64         `json:"train_steps"`
	WeightKiB  float64       `json:"weight_kib"`
	MeanLoss   float64       `json:"mean_loss"`
	LastLoss   float64       `json:"last_loss"`
	Fit        SplitScore    `json:"fit"`   // reconstruction of demo pairs
	Train      SplitScore    `json:"train"` // training-split test grids
	Eval       SplitScore    `json:"eval"`  // evaluation-split test grids
	Lucy       lucy.Snapshot `json:"lucy"`
	Err        string        `json:"error,omitempty"`
}

type RunResults struct {
	Engine       string                 `json:"engine"`
	Layer        string                 `json:"layer"`
	Mode         string                 `json:"mode"`
	Camerals     int                    `json:"camerals"`
	Hidden       int                    `json:"hidden"`
	ItemTime     string                 `json:"item_time"`
	Passes       int                    `json:"passes"`
	TrainDir     string                 `json:"train_dir"`
	EvalDir      string                 `json:"eval_dir"`
	TrainN       int                    `json:"train_tasks"`
	EvalN        int                    `json:"eval_tasks"`
	TrainItems   int                    `json:"train_items"`
	Timestamp    string                 `json:"timestamp"`
	Duration     string                 `json:"duration"`
	Jobs         []string               `json:"jobs"`
	Results      map[string]*ArchResult `json:"results"`
	ScoreFormula string                 `json:"score_formula"`
}

type encodedPair struct {
	taskID string
	x, y   *core.Tensor[float32]
	gold   [][]int
}

func main() {
	nTiles := flag.Int("n", 0, "cap tasks per split (0 = all training + all eval)")
	offset := flag.Int("offset", 0, "skip first N tasks in each split")
	camerals := flag.Int("camerals", 5, "max native hemispheres (sweeps cam-min..N plus Dense)")
	camMin := flag.Int("cam-min", 2, "first cameral count (inclusive)")
	layerName := flag.String("layer", "dense", "cell kind (dense now; others reserved)")
	hidden := flag.Int("hidden", 64, "hidden width")
	itemTime := flag.Duration("item-time", 125*time.Millisecond, "TrainStackMSE budget per training demo")
	passes := flag.Int("passes", 1, "cycles over the full training demo set")
	lr := flag.Float64("lr", defaultLR, "TrainStackMSE learning rate")
	workers := flag.Int("workers", 0, "concurrent cameral jobs (0 = all arches / NumCPU)")
	seed := flag.Int64("seed", 1, "rng seed")
	set := flag.String("set", "agi1", "agi1 | agi2")
	dataDir := flag.String("data", "", "override ARC data root (expects training/ and evaluation/)")
	noDense := flag.Bool("no-dense", false, "skip Dense baseline")
	flag.Parse()

	kind, err := parseCellKind(*layerName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(2)
	}
	if *camerals < 1 {
		*camerals = 1
	}
	if *camMin < 2 {
		*camMin = 2
	}
	if *camMin > *camerals {
		*camMin = *camerals
	}
	if *hidden < 8 {
		*hidden = 8
	}
	if *passes < 1 {
		*passes = 1
	}
	if *itemTime <= 0 {
		*itemTime = 125 * time.Millisecond
	}

	trainDir, evalDir := splitDirs(*set, *dataDir)
	rand.Seed(*seed)

	trainTasks, err := loadTasks(trainDir, *offset, *nTiles)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ load training: %v\n", err)
		os.Exit(1)
	}
	evalTasks, err := loadTasks(evalDir, *offset, *nTiles)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ load evaluation: %v\n", err)
		os.Exit(1)
	}
	demos := flattenDemos(trainTasks)

	mode := parallel.ModeStepTweenChain
	dim := vecDim(MaxGrid)
	jobs := buildJobs(*camMin, *camerals, !*noDense)
	if *workers <= 0 {
		*workers = len(jobs)
		if n := runtime.NumCPU(); n < *workers {
			*workers = n
		}
		if *workers < 1 {
			*workers = 1
		}
	}

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   TEST 44 — train ALL ARC demos (125ms/item) then solve-train + eval split          ║")
	fmt.Println("║   Lucy/tide: SoftAcc AdaptPct Avail Score MobileScore  |  welvet v0.95.1 camerals    ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("\n📂 train %s\n📂 eval  %s\n", trainDir, evalDir)
	fmt.Printf("🧩 train tasks=%d  demos=%d  eval tasks=%d\n", len(trainTasks), len(demos), len(evalTasks))
	fmt.Printf("🧠")
	if !*noDense {
		fmt.Printf(" Dense")
	}
	for n := *camMin; n <= *camerals; n++ {
		fmt.Printf(" + %s", camName(n))
	}
	fmt.Printf("  → %d nets  workers=%d  cpus=%d\n", len(jobs), *workers, runtime.NumCPU())
	fmt.Printf("⏱️  %s/demo × %d pass(es)  (~%s train/net)  layer=%s hidden=%d dim=%d lr=%.4f  mode=%s\n\n",
		*itemTime, *passes, time.Duration(len(demos)**passes)**itemTime, kind, *hidden, dim, *lr, mode)

	results := &RunResults{
		Engine:       "welvet-v0.95.1-native-cam",
		Layer:        string(kind),
		Mode:         mode.String(),
		Camerals:     *camerals,
		Hidden:       *hidden,
		ItemTime:     itemTime.String(),
		Passes:       *passes,
		TrainDir:     trainDir,
		EvalDir:      evalDir,
		TrainN:       len(trainTasks),
		EvalN:        len(evalTasks),
		TrainItems:   len(demos),
		Timestamp:    time.Now().Format(time.RFC3339),
		Jobs:         make([]string, len(jobs)),
		Results:      make(map[string]*ArchResult, len(jobs)),
		ScoreFormula: "Throughput × Availability × SoftAcc / 10000",
	}
	for i, j := range jobs {
		results.Jobs[i] = camName(j.nHemi)
	}

	start := time.Now()
	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, *workers)
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			name := camName(j.nHemi)
			r := runArch(j, kind, *hidden, *itemTime, *passes, *lr, mode, demos, trainTasks, evalTasks)
			mu.Lock()
			results.Results[name] = r
			mu.Unlock()
			if r.Err != "" {
				fmt.Printf("❌ [%s] %s\n", name, r.Err)
				return
			}
			fmt.Printf("✅ [%s] fit:%.0f%% train:%d/%d eval:%d/%d  soft=%.1f adapt=%.1f avail=%.1f score=%.0f mobile=%.0f\n",
				name, r.Fit.MeanPixel, r.Train.Solved, r.Train.Tasks, r.Eval.Solved, r.Eval.Tasks,
				r.Lucy.SoftAcc, r.Lucy.AdaptPct, r.Lucy.Availability, r.Lucy.Score, r.Lucy.MobileScore)
		}(j)
	}
	wg.Wait()
	results.Duration = time.Since(start).String()
	saveResults(results)
	printSummary(results)
}

func splitDirs(set, override string) (trainDir, evalDir string) {
	root := override
	if root == "" {
		switch strings.ToLower(set) {
		case "agi2", "arc-agi2", "2":
			root = filepath.Join("..", "ARC-AGI2", "data")
		default:
			root = filepath.Join("..", "ARC-AGI", "data")
		}
	}
	base := filepath.Base(root)
	switch strings.ToLower(base) {
	case "training", "evaluation":
		root = filepath.Dir(root)
	}
	return filepath.Join(root, "training"), filepath.Join(root, "evaluation")
}

func buildJobs(camMin, camMax int, includeDense bool) []job {
	var out []job
	if includeDense {
		out = append(out, job{nHemi: 1})
	}
	for n := camMin; n <= camMax; n++ {
		out = append(out, job{nHemi: n})
	}
	return out
}

func flattenDemos(tasks []Task) []encodedPair {
	var out []encodedPair
	for _, t := range tasks {
		for _, p := range t.Train {
			out = append(out, encodedPair{
				taskID: t.ID,
				x:      inputTensor(encodeGrid(p.Input, MaxGrid)),
				y:      inputTensor(encodeGrid(p.Output, MaxGrid)),
				gold:   p.Output,
			})
		}
	}
	return out
}

func runArch(j job, kind CellKind, hidden int, itemTime time.Duration, passes int, lr float64, mode parallel.TrainMode, demos []encodedPair, trainTasks, evalTasks []Task) *ArchResult {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	r := &ArchResult{
		Label:    camName(j.nHemi),
		Arch:     camName(j.nHemi),
		Hemis:    j.nHemi,
		Layer:    string(kind),
		Mode:     mode.String(),
		Items:    len(demos),
		Passes:   passes,
		ItemTime: itemTime.String(),
	}

	stack, err := buildNativeCameral(kind, vecDim(MaxGrid), hidden, vecDim(MaxGrid), j.nHemi, mode)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.Lucy.WeightBytes = stackWeightBytes(stack)
	r.WeightKiB = float64(r.Lucy.WeightBytes) / 1024
	heapMu.Lock()
	before := heapNow()
	heapMu.Unlock()
	fmt.Printf("🚀 [%s] cycling %d demos  weights=%.1f KiB  hemis=%d\n", r.Arch, len(demos), r.WeightKiB, j.nHemi)

	wall0 := time.Now()
	var totalTrain, totalInfer time.Duration
	var lastLoss, lossSum, softSum, pixSum float64
	var hit25, hit50 bool
	total := len(demos) * passes
	done := 0
	logEvery := 50
	if total >= 500 {
		logEvery = 100
	}
	const rollN = 100
	roll := make([]float64, 0, rollN)
	rollSoft := make([]float64, 0, rollN)
	prevTask := ""

	for pass := 0; pass < passes; pass++ {
		for i, item := range demos {
			deadline := time.Now().Add(itemTime)
			steps := 0
			var itemTrain time.Duration
			for {
				t0 := startWork()
				loss, terr := parallel.TrainStackMSE(stack, item.x, item.y, mode, lr)
				d := t0.elapsed()
				itemTrain += d
				totalTrain += d
				if terr != nil {
					r.Err = fmt.Sprintf("train %s[%d]: %v", item.taskID, i, terr)
					return r
				}
				lastLoss = loss
				r.TrainSteps++
				r.Lucy.TotalTrain++
				steps++
				if !time.Now().Before(deadline) {
					break
				}
			}

			tInf := startWork()
			_, post, ferr := parallel.ForwardStack(stack, item.x)
			inferDur := tInf.elapsed()
			totalInfer += inferDur

			soft, pix := 0.0, 0.0
			var correct int64
			if ferr == nil && post != nil {
				r.Lucy.TotalOutputs++
				soft = colorSoftAcc(post.Data, item.y.Data, MaxGrid)
				ph, pw := decodeSize(post.Data, MaxGrid)
				pred := decodeGrid(post.Data, ph, pw, MaxGrid)
				pix = pixelAcc(pred, item.gold)
				if gridsEqual(pred, item.gold) {
					correct = 1
					r.Lucy.TotalCorrect++
				}
			}
			softSum += soft
			pixSum += pix
			lossSum += lastLoss
			roll = append(roll, lastLoss)
			rollSoft = append(rollSoft, soft)
			if len(roll) > rollN {
				roll = roll[len(roll)-rollN:]
				rollSoft = rollSoft[len(rollSoft)-rollN:]
			}

			sw := 0
			if item.taskID != prevTask {
				if prevTask != "" {
					sw = 1
				}
				prevTask = item.taskID
			}
			w := lucy.Window{
				At:            time.Now(),
				Outputs:       1,
				Correct:       correct,
				TrainSteps:    int64(steps),
				InferMs:       inferDur.Seconds() * 1000,
				TrainMs:       itemTrain.Seconds() * 1000,
				Phase:         item.taskID,
				PhaseSwitches: sw,
				Accuracy:      pix,
				SoftAcc:       soft,
			}
			r.Lucy.Windows = lucy.AppendWindow(r.Lucy.Windows, w)
			r.Lucy.SoftAccBlocks = append(r.Lucy.SoftAccBlocks, soft)
			r.Lucy.PhaseBlocks = append(r.Lucy.PhaseBlocks, item.taskID)
			r.Lucy.SwitchBlocks = append(r.Lucy.SwitchBlocks, sw > 0)
			r.Lucy.AccuracyPulses++

			elapsed := time.Since(wall0).Seconds()
			if !hit25 && pix >= lucy.AccThreshold25 {
				r.Lucy.TimeToAcc25Sec = elapsed
				hit25 = true
			}
			if !hit50 && pix >= lucy.AccThreshold50 {
				r.Lucy.TimeToAcc50Sec = elapsed
				hit50 = true
			}

			done++
			if done%logEvery == 0 || done == total {
				fmt.Printf("   [%s] %d/%d  last=%s  steps=%d  item=%.4f  mean100=%.4f  soft=%.1f  pix=%.1f\n",
					r.Arch, done, total, item.taskID, steps, lastLoss, meanF(roll), meanF(rollSoft), pix)
			}
		}
	}
	r.LastLoss = lastLoss
	if done > 0 {
		r.MeanLoss = lossSum / float64(done)
		r.Lucy.SoftAcc = softSum / float64(done)
		r.Lucy.AvgAccuracy = pixSum / float64(done)
	}

	r.Fit = scoreEncoded(stack, demos, "fit", &totalInfer, &r.Lucy)
	r.Train = scoreTasks(stack, trainTasks, "train", &totalInfer, &r.Lucy)
	r.Eval = scoreTasks(stack, evalTasks, "eval", &totalInfer, &r.Lucy)

	heapMu.Lock()
	after := heapNow()
	heapMu.Unlock()
	r.Lucy.HeapBytes = int64(after - before)
	if r.Lucy.HeapBytes < 0 {
		r.Lucy.HeapBytes = 0
	}
	r.Lucy.Duration = time.Since(wall0)
	r.Lucy.InferMs = totalInfer.Seconds() * 1000
	r.Lucy.TrainMs = totalTrain.Seconds() * 1000
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: lucy.AdaptWindowsDefault, ConsThreshold: lucy.ConsThreshold})
	return r
}

func scoreEncoded(stack *parallel.Stack, items []encodedPair, name string, infer *time.Duration, snap *lucy.Snapshot) SplitScore {
	s := SplitScore{Name: name, Tasks: len(items), TestGrids: len(items)}
	if len(items) == 0 {
		return s
	}
	var pix, soft float64
	byTaskExact := map[string]bool{}
	byTaskSeen := map[string]bool{}
	for _, item := range items {
		tInf := startWork()
		_, post, err := parallel.ForwardStack(stack, item.x)
		*infer += tInf.elapsed()
		if err != nil || post == nil {
			continue
		}
		snap.TotalOutputs++
		goldY := item.y.Data
		ph, pw := decodeSize(post.Data, MaxGrid)
		pred := decodeGrid(post.Data, ph, pw, MaxGrid)
		acc := pixelAcc(pred, item.gold)
		sa := colorSoftAcc(post.Data, goldY, MaxGrid)
		pix += acc
		soft += sa
		exact := gridsEqual(pred, item.gold)
		if exact {
			s.ExactGrids++
		}
		if prev, ok := byTaskExact[item.taskID]; !ok {
			byTaskExact[item.taskID] = exact
			byTaskSeen[item.taskID] = true
		} else {
			byTaskExact[item.taskID] = prev && exact
		}
	}
	s.MeanPixel = pix / float64(len(items))
	s.MeanSoft = soft / float64(len(items))
	s.Tasks = len(byTaskSeen)
	for id, ok := range byTaskExact {
		if ok {
			s.Solved++
			s.SolvedIDs = append(s.SolvedIDs, id)
		}
	}
	return s
}

func scoreTasks(stack *parallel.Stack, tasks []Task, name string, infer *time.Duration, snap *lucy.Snapshot) SplitScore {
	s := SplitScore{Name: name, Tasks: len(tasks)}
	if len(tasks) == 0 {
		return s
	}
	var pix, soft float64
	var nGrid int
	for _, t := range tasks {
		allExact := len(t.Test) > 0
		for _, p := range t.Test {
			x := inputTensor(encodeGrid(p.Input, MaxGrid))
			y := encodeGrid(p.Output, MaxGrid)
			tInf := startWork()
			_, post, err := parallel.ForwardStack(stack, x)
			*infer += tInf.elapsed()
			if err != nil || post == nil {
				allExact = false
				continue
			}
			snap.TotalOutputs++
			nGrid++
			ph, pw := decodeSize(post.Data, MaxGrid)
			pred := decodeGrid(post.Data, ph, pw, MaxGrid)
			pix += pixelAcc(pred, p.Output)
			soft += colorSoftAcc(post.Data, y, MaxGrid)
			exact := gridsEqual(pred, p.Output)
			if exact {
				s.ExactGrids++
			} else {
				allExact = false
			}
		}
		if allExact {
			s.Solved++
			s.SolvedIDs = append(s.SolvedIDs, t.ID)
		}
	}
	s.TestGrids = nGrid
	if nGrid > 0 {
		s.MeanPixel = pix / float64(nGrid)
		s.MeanSoft = soft / float64(nGrid)
	}
	return s
}

func meanF(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func saveResults(results *RunResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	_ = os.WriteFile("test44_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test44_results.json")
}

func printSummary(results *RunResults) {
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  ARC — fit demos / solve training tests / eval split (exact = official grid match)                       ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  %-14s │ KiB  │ FitPix │ TrainPix │ TrainSolve │ EvalPix │ EvalSolve │ Steps    │ MeanLoss ║\n", "Arch")
	fmt.Println("║  ──────────────┼──────┼────────┼──────────┼────────────┼─────────┼───────────┼──────────┼──────────║")
	for _, name := range results.Jobs {
		r := results.Results[name]
		if r == nil {
			continue
		}
		if r.Err != "" {
			fmt.Printf("║  %-14s │ ERR %s\n", name, r.Err)
			continue
		}
		fmt.Printf("║  %-14s │ %4.0f │ %5.1f%% │  %5.1f%%  │ %4d/%-4d  │ %5.1f%%  │ %4d/%-4d │ %8d │ %8.4f ║\n",
			name, r.WeightKiB, r.Fit.MeanPixel, r.Train.MeanPixel,
			r.Train.Solved, r.Train.Tasks,
			r.Eval.MeanPixel, r.Eval.Solved, r.Eval.Tasks,
			r.TrainSteps, r.MeanLoss)
	}

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  LUCY / tide — Score = T × Availability × SoftAcc / 10_000   |   MobileScore = Score / WeightMiB          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Arch           │ Soft   │ Adapt  │ Avail  │ Stab   │ Cons   │ Tput     │ Score    │ Mobile  │ ZDT    ║")
	fmt.Println("║  ───────────────┼────────┼────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┼────────║")
	best, bestName := -1.0, ""
	bestMob, bestMobName := -1.0, ""
	bestRAM, bestRAMName := int64(1<<62), ""
	for _, name := range results.Jobs {
		r := results.Results[name]
		if r == nil || r.Err != "" {
			continue
		}
		s := r.Lucy
		fmt.Printf("║  %-14s │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %8.0f │ %8.0f │ %7.0f │ %6.1f ║\n",
			name, s.SoftAcc, s.AdaptPct, s.Availability, s.Stability, s.Consistency,
			s.Throughput, s.Score, s.MobileScore, s.ZeroDowntime)
		if s.Score > best {
			best, bestName = s.Score, name
		}
		if s.MobileScore > bestMob {
			bestMob, bestMobName = s.MobileScore, name
		}
		if s.WeightBytes > 0 && s.WeightBytes < bestRAM {
			bestRAM, bestRAMName = s.WeightBytes, name
		}
	}
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  InferMs/TrainMs + Acc/sec:\n")
	for _, name := range results.Jobs {
		r := results.Results[name]
		if r == nil || r.Err != "" {
			continue
		}
		s := r.Lucy
		fmt.Printf("║    %-14s  infer=%.0fms train=%.0fms  t25=%.1fs t50=%.1fs  acc/s=%.3f  heap=%.1fKiB\n",
			name, s.InferMs, s.TrainMs, s.TimeToAcc25Sec, s.TimeToAcc50Sec, s.AccPerSec, s.HeapMiB*1024)
	}
	fmt.Printf("║  🏆 WINNER (Score):        %-14s  %.0f\n", bestName, best)
	fmt.Printf("║  📱 Best mobile Score/MiB: %-14s  %.0f\n", bestMobName, bestMob)
	fmt.Printf("║  💾 Smallest weights:      %-14s  %.1f KiB\n", bestRAMName, float64(bestRAM)/1024)
	fmt.Printf("║  train tasks=%d demos=%d  eval tasks=%d  item-time=%s  passes=%d  duration=%s\n",
		results.TrainN, results.TrainItems, results.EvalN, results.ItemTime, results.Passes, results.Duration)
	fmt.Printf("║  layer=%s  mode=%s  camerals≤%d  hidden=%d  duty=%s\n",
		results.Layer, results.Mode, results.Camerals, results.Hidden, dutyClockName())
	fmt.Println("║  AdaptPct = mean SoftAcc on first 4 demos after each task switch (tide AdaptWindows).")
	fmt.Println("║  Availability = InferMs/(InferMs+TrainMs)×100  |  MobileScore = Score/WeightMiB")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

var heapMu sync.Mutex

func heapNow() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func stackWeightBytes(s *parallel.Stack) int64 {
	var n int64
	for _, st := range dna.CollectStores(s) {
		n += storeBytes(st)
	}
	return n
}

func storeBytes(s *weights.Store) int64 {
	if s == nil {
		return 0
	}
	n := int64(len(s.Bias) * 8)
	if s.Packed != nil {
		n += int64(len(s.Packed.Raw))
		n += int64(len(s.Packed.Scales) * 4)
		n += int64(len(s.Packed.Mins) * 4)
		n += int64(len(s.Packed.Meta))
		n += int64(len(s.Packed.Q4Packed) * 4)
		n += int64(len(s.Packed.Int8QS))
		n += int64(len(s.Packed.F32Cache) * 4)
		return n
	}
	if len(s.Native) > 0 {
		return n + int64(len(s.Native))
	}
	bits := s.DType.Bits()
	if bits <= 0 {
		bits = 32
	}
	return n + int64((s.Rows*s.Cols*bits+7)/8)
}
