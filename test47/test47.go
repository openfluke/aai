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
)

// TEST 47 — Tween / Split / Alt on normal toys.
//
// TweenChain is backprop (BackwardStack+SGD). These are not:
//   Tween / StepTween           — broadcast output gap onto every leaf (full gap, half LR)
//   TweenSplit / StepTweenSplit — same gap, split 1/N across leaves
//   TweenAlt / StepTweenAlt     — Split then Tween, repeat AltTimes (recompute MSE gap)
// On a Sandwich, Step* and non-Step share a family (same update). Columns still
// run as separate jobs so we can see if they match.

const defaultLR = 0.05

type sample struct {
	x, y *core.Tensor[float32]
}

type toyTask struct {
	name   string
	in     int
	out    int
	train  []sample
	eval   []sample
	binary bool // threshold 0.5 exact-match; else |err|<0.15
}

type job struct {
	task  toyTask
	kind  CellKind
	nHemi int
	mode  parallel.TrainMode
}

type row struct {
	Task    string  `json:"task"`
	Layer   string  `json:"layer"`
	Arch    string  `json:"arch"`
	Mode    string  `json:"mode"`
	Acc     float64 `json:"acc"`
	Soft    float64 `json:"soft_acc"`
	Loss    float64 `json:"loss"`
	Steps   int64   `json:"steps"`
	Broke50 bool    `json:"broke_50"`
	Err     string  `json:"error,omitempty"`
}

func main() {
	layersFlag := flag.String("layers", "all", "comma list, or all (including dense)")
	camerals := flag.Int("camerals", 2, "max hemispheres (cam-min..N)")
	camMin := flag.Int("cam-min", 1, "first cameral count")
	only := flag.Int("only", 0, "exactly this many hemispheres")
	hidden := flag.Int("hidden", 32, "hidden width")
	budget := flag.Duration("budget", 1500*time.Millisecond, "TrainStackMSE wall per job")
	lr := flag.Float64("lr", defaultLR, "learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	workers := flag.Int("workers", 0, "concurrent jobs (0 = NumCPU)")
	tasksFlag := flag.String("tasks", "xor,sine,copy", "xor,sine,copy")
	modesFlag := flag.String("modes", "steptween,tween,tweensplit,steptweensplit,tweenalt,steptweenalt",
		"steptween,tween,tweensplit,steptweensplit,tweenalt,steptweenalt  (add chain for backprop)")
	altTimes := flag.Int("alt-times", 1, "TweenAlt: Split→Tween pairs per TrainStackMSE call")
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
	if *only > 0 {
		*camMin, *camerals = *only, *only
	}
	if *camMin < 1 {
		*camMin = 1
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
	rand.Seed(*seed)

	tasks := buildTasks(*tasksFlag, *seed)
	if len(tasks) == 0 {
		fmt.Fprintf(os.Stderr, "❌ no tasks\n")
		os.Exit(2)
	}

	var jobs []job
	for _, t := range tasks {
		for _, k := range kinds {
			for n := *camMin; n <= *camerals; n++ {
				for _, m := range modes {
					jobs = append(jobs, job{task: t, kind: k, nHemi: n, mode: m})
				}
			}
		}
	}

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   TEST 47 — Tween | Split | Alt (Split↔Tween ping-pong)                             ║")
	fmt.Println("║   TweenChain = backprop. Split = 1/N. AltTimes = Split→Tween pairs per sample.      ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("🧠 layers=%s\n", kindsCSV(kinds))
	fmt.Printf("📐 cams=%d..%d  hidden=%d  budget=%s/job  lr=%.3f  alt-times=%d  jobs=%d  workers=%d\n",
		*camMin, *camerals, *hidden, *budget, *lr, *altTimes, len(jobs), *workers)
	fmt.Printf("🧪 tasks=%s  modes=%s\n\n", *tasksFlag, modesCSV(modes))

	rows := make([]row, len(jobs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)
	for i, j := range jobs {
		wg.Add(1)
		go func(i int, j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			rows[i] = runJob(j, *hidden, *budget, *lr, *altTimes)
			r := rows[i]
			tag := fmt.Sprintf("%s %s/%s %s", r.Task, r.Layer, r.Arch, r.Mode)
			if r.Err != "" {
				fmt.Printf("❌ [%s] %s\n", tag, r.Err)
				return
			}
			mark := "  "
			if r.Broke50 {
				mark = "💥"
			}
			fmt.Printf("%s [%s] acc=%.1f%% soft=%.1f loss=%.4f steps=%d\n",
				mark, tag, r.Acc, r.Soft, r.Loss, r.Steps)
		}(i, j)
	}
	wg.Wait()

	data, _ := json.MarshalIndent(map[string]any{
		"engine":    "welvet-tween-vs-split",
		"alt_times": *altTimes,
		"rows":      rows,
	}, "", "  ")
	_ = os.WriteFile("test47_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test47_results.json")
	printTables(rows, tasks, modes)
}

func parseModes(s string) ([]parallel.TrainMode, error) {
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, p := range splitLayerTokens(s) {
		var m parallel.TrainMode
		switch strings.ToLower(p) {
		case "tween":
			m = parallel.ModeTween
		case "steptween":
			m = parallel.ModeStepTween
		case "tweensplit", "split":
			m = parallel.ModeTweenSplit
		case "steptweensplit", "stepsplit":
			m = parallel.ModeStepTweenSplit
		case "chain", "tweenchain", "steptweenchain", "bp":
			m = parallel.ModeStepTweenChain
		case "alt", "tweenalt":
			m = parallel.ModeTweenAlt
		case "steptweenalt", "stepalt":
			m = parallel.ModeStepTweenAlt
		default:
			return nil, fmt.Errorf("unknown mode %q (tween|steptween|tweensplit|steptweensplit|tweenalt|steptweenalt|chain)", p)
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
			out = append(out, makeSine(32))
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
	return toyTask{name: "xor", in: dim, out: 1, train: set, eval: set, binary: true}
}

func makeSine(n int) toyTask {
	const dim = 16
	var set []sample
	for i := 0; i < n; i++ {
		x := 2 * math.Pi * float64(i) / float64(n)
		in := vec(dim)
		in.Data[0] = float32(x / (2 * math.Pi))
		in.Data[1] = float32(math.Sin(x))
		in.Data[2] = float32(math.Cos(x))
		y := float32((math.Sin(x) + 1) / 2)
		set = append(set, sample{x: in, y: vec(1, y)})
	}
	return toyTask{name: "sine", in: dim, out: 1, train: set, eval: set, binary: false}
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
	return toyTask{name: "copy", in: dim, out: dim, train: train, eval: eval, binary: true}
}

func runJob(j job, hidden int, budget time.Duration, lr float64, altTimes int) row {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r := row{
		Task:  j.task.name,
		Layer: string(j.kind),
		Arch:  camName(j.nHemi),
		Mode:  j.mode.String(),
	}
	stack, err := buildNativeCameral(j.kind, j.task.in, hidden, j.task.out, j.nHemi, j.mode)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	stack.AltTimes = altTimes
	deadline := time.Now().Add(budget)
	var last float64
	for time.Now().Before(deadline) {
		for _, s := range j.task.train {
			if !time.Now().Before(deadline) {
				break
			}
			loss, terr := parallel.TrainStackMSE(stack, s.x, s.y, j.mode, lr)
			if terr != nil {
				r.Err = terr.Error()
				return r
			}
			last = loss
			r.Steps++
		}
	}
	r.Loss = last
	r.Acc, r.Soft = scoreTask(stack, j.task)
	r.Broke50 = r.Acc > 50+1e-9
	return r
}

func scoreTask(stack *parallel.Stack, task toyTask) (acc, soft float64) {
	if len(task.eval) == 0 {
		return 0, 0
	}
	var hit, bits, okBits float64
	var softSum float64
	nSoft := 0
	for _, s := range task.eval {
		_, post, err := parallel.ForwardStack(stack, s.x)
		if err != nil || post == nil {
			continue
		}
		n := post.Len()
		if s.y.Len() < n {
			n = s.y.Len()
		}
		all := true
		for i := 0; i < n; i++ {
			p := float64(post.Data[i])
			t := float64(s.y.Data[i])
			if task.binary {
				pred := 0.0
				if p >= 0.5 {
					pred = 1
				}
				gold := 0.0
				if t >= 0.5 {
					gold = 1
				}
				bits++
				if pred == gold {
					okBits++
				} else {
					all = false
				}
				softSum += lucy.SoftAccProb(post.Data[i], s.y.Data[i])
			} else {
				if math.Abs(p-t) > 0.15 {
					all = false
				}
				softSum += lucy.SoftAccOne(post.Data[i], s.y.Data[i])
			}
			nSoft++
		}
		if all {
			hit++
		}
	}
	if task.binary && bits > 0 && task.name == "copy" {
		acc = 100 * okBits / bits
	} else {
		acc = 100 * hit / float64(len(task.eval))
	}
	if nSoft > 0 {
		soft = softSum / float64(nSoft)
	}
	return acc, soft
}

func printTables(rows []row, tasks []toyTask, modes []parallel.TrainMode) {
	byTask := map[string][]row{}
	for _, r := range rows {
		byTask[r.Task] = append(byTask[r.Task], r)
	}
	for _, t := range tasks {
		rs := byTask[t.name]
		fmt.Println()
		fmt.Printf("══ %s  in=%d out=%d  eval=%d  (💥 = Acc > 50%%) ══\n", t.name, t.in, t.out, len(t.eval))
		fmt.Printf("%-16s %-12s", "layer/arch", "")
		for _, m := range modes {
			fmt.Printf(" │ %-16s", m.String())
		}
		fmt.Println()
		fmt.Printf("%-16s %-12s", "", "")
		for range modes {
			fmt.Printf(" │ %-5s %-5s %-5s", "Acc", "Soft", "Loss")
		}
		fmt.Println()

		type key struct{ layer, arch string }
		grouped := map[key]map[string]row{}
		var keys []key
		seen := map[key]bool{}
		for _, r := range rs {
			k := key{r.Layer, r.Arch}
			if grouped[k] == nil {
				grouped[k] = map[string]row{}
			}
			grouped[k][r.Mode] = r
			if !seen[k] {
				seen[k] = true
				keys = append(keys, k)
			}
		}
		sort.Slice(keys, func(i, j int) bool {
			if keys[i].layer != keys[j].layer {
				return keys[i].layer < keys[j].layer
			}
			return keys[i].arch < keys[j].arch
		})
		best := -1.0
		bestName := ""
		broke := 0
		for _, k := range keys {
			fmt.Printf("%-16s %-12s", k.layer, k.arch)
			for _, m := range modes {
				r, ok := grouped[k][m.String()]
				if !ok {
					fmt.Printf(" │ %-16s", "—")
					continue
				}
				if r.Err != "" {
					fmt.Printf(" │ ERR %-12s", trimErr(r.Err, 12))
					continue
				}
				mark := " "
				if r.Broke50 {
					mark = "💥"
					broke++
				}
				fmt.Printf(" │%s%5.1f %5.1f %5.3f", mark, r.Acc, r.Soft, r.Loss)
				if r.Acc > best {
					best, bestName = r.Acc, fmt.Sprintf("%s/%s %s", k.layer, k.arch, r.Mode)
				}
			}
			fmt.Println()
		}
		fmt.Printf("   broke 50%%: %d rows    best Acc: %s  %.1f%%\n", broke, bestName, best)
	}
}

func trimErr(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n]
	}
	return s
}
