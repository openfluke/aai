package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
)

type modeResult struct {
	ID         string        `json:"id,omitempty"`
	Mode       string        `json:"mode"`
	Layer      string        `json:"layer,omitempty"`
	DType      string        `json:"dtype,omitempty"`
	LR         float64       `json:"lr,omitempty"`
	Challenge  string        `json:"challenge,omitempty"`
	Cams       int           `json:"cams,omitempty"`
	GridN      int           `json:"grid_n,omitempty"`
	Phase      string        `json:"phase,omitempty"`
	Acc        float64       `json:"acc"`
	Soft       float64       `json:"soft_acc"`
	Avail      float64       `json:"availability"`
	Thru       float64       `json:"throughput"`
	Score      float64       `json:"score"`
	RAMKiB     float64       `json:"ram_kib"`
	Levels     int           `json:"levels"`
	Actions    int64         `json:"actions"`
	ThinkK     int           `json:"think_k"`
	AccDelta   float64       `json:"acc_delta,omitempty"`
	ScoreDelta float64       `json:"score_delta,omitempty"`
	ImprovePct float64       `json:"improve_pct,omitempty"` // AccΔ relative to train Acc
	Promoted   bool          `json:"promoted,omitempty"`
	Err        string        `json:"error,omitempty"`
	Lucy       lucy.Snapshot `json:"lucy"`
	Weights    [][]float32   `json:"-"`
}

type runCfg struct {
	mode      parallel.TrainMode
	initW     [][]float32
	dur       time.Duration
	win       time.Duration
	lr        float64
	seed      int64
	thinkK    int
	game      string
	challenge string
	bridgePy  string
	layer     string
	dtype     core.DType
	cams      int
	gridN     int
	train     bool // false = freeze weights (think+act only)
	phase     string
	jobID     string
	hub       *liveHub
	promoted  bool
}


func summarizeTree(t Tree, leaves []modeResult) treeReport {
	rep := treeReport{
		Key: t.Key, Mode: t.Mode, Layer: t.Layer, DType: t.DType, Challenge: t.Challenge,
		Leaves: len(t.Jobs), Finished: time.Now().UTC(),
	}
	best := -1.0
	for _, r := range leaves {
		row := leafRow{
			ID: r.ID, LR: r.LR, Cams: r.Cams, GridN: r.GridN, Phase: r.Phase,
			Acc: r.Acc, Soft: r.Soft, Avail: r.Avail, Thru: r.Thru, Score: r.Score, RAMKiB: r.RAMKiB,
			Levels: r.Levels, AccΔ: r.AccDelta, Improve: r.ImprovePct, Done: true, Err: r.Err,
		}
		rep.Rows = append(rep.Rows, row)
		if r.Err != "" {
			continue
		}
		if r.Phase != "" && r.Phase != "train" && r.Phase != "after_train" {
			continue
		}
		score := r.Score
		if score > best || (score == best && r.Acc > rep.BestAcc) {
			best = score
			rep.BestID = r.ID
			rep.BestAcc = r.Acc
			rep.BestScore = r.Score
			rep.BestΔ = r.AccDelta
			rep.BestLR = r.LR
			rep.BestCams = r.Cams
			rep.BestGrid = r.GridN
		}
	}
	return rep
}


func baseJobID(id string) string {
	for _, suf := range []string{"|after_train", "|after_freeze", "|freeze", "|after", "|promote"} {
		if len(id) > len(suf) && id[len(id)-len(suf):] == suf {
			return id[:len(id)-len(suf)]
		}
	}
	return id
}

func phaseRank(phase string) int {
	switch phase {
	case "after_train":
		return 3
	case "train", "":
		return 2
	case "after_freeze", "freeze":
		return 1
	default:
		return 0
	}
}

// rebuildReportsFromResults rebuilds consolidation reports for fully-done trees from ckpt rows.
func rebuildReportsFromResults(trees []Tree, results []modeResult, done map[string]bool) []treeReport {
	byID := map[string]modeResult{}
	for _, r := range results {
		id := baseJobID(r.ID)
		prev, ok := byID[id]
		if !ok || phaseRank(r.Phase) >= phaseRank(prev.Phase) {
			rr := r
			rr.ID = id
			byID[id] = rr
		}
	}
	var out []treeReport
	for _, tree := range trees {
		allDone := true
		for _, j := range tree.Jobs {
			if !done[j.ID] {
				allDone = false
				break
			}
		}
		if !allDone {
			continue
		}
		leaves := make([]modeResult, 0, len(tree.Jobs))
		for _, j := range tree.Jobs {
			if r, ok := byID[j.ID]; ok {
				leaves = append(leaves, r)
			}
		}
		if len(leaves) == 0 {
			continue
		}
		out = append(out, summarizeTree(tree, leaves))
	}
	return out
}

// enrichReportsFromResults fills missing throughput on saved report rows from results.json.
// Older reports were saved before throughput was copied into leaf rows — no re-run needed.
func enrichReportsFromResults(reports []treeReport, results []modeResult) bool {
	byID := map[string]modeResult{}
	for _, r := range results {
		id := baseJobID(r.ID)
		prev, ok := byID[id]
		if !ok || phaseRank(r.Phase) >= phaseRank(prev.Phase) {
			rr := r
			rr.ID = id
			byID[id] = rr
		}
	}
	changed := false
	for ri := range reports {
		for li := range reports[ri].Rows {
			row := &reports[ri].Rows[li]
			m, ok := byID[row.ID]
			if !ok {
				continue
			}
			if row.Thru == 0 && m.Thru > 0 {
				row.Thru = m.Thru
				changed = true
			}
		}
	}
	return changed
}

func main() {
	modesFlag := flag.String("modes", "all", "all = AllNamedTrainModes, or csv (sgd,step_sgd,TweenSplitSparse,…)")
	thinkK := flag.Int("think", 4, "recurrent self-think steps before each env action")
	dur := flag.Duration("duration", 3*time.Second, "wall per job train phase")
	afterFreeze := flag.Duration("after-freeze", 2*time.Second, "after train: keep thinking with frozen weights")
	afterTrain := flag.Duration("after-train", 3*time.Second, "after freeze: keep training (self-improve past stop)")
	promote := flag.Duration("promote", 5*time.Second, "extra wall for LPD champ after sweep")
	win := flag.Duration("window", time.Second, "Lucy pulse window")
	lr := flag.Float64("lr", 0.05, "single LR when -lrs empty (ignored if -lrs set)")
	lrsFlag := flag.String("lrs", "funny", "funny|all = 0.02…1e6 sweep, or csv; empty = use -lr once")
	layersFlag := flag.String("layers", "all", "dense|dense-wide|dense-deep|dense-deep-wide|all")
	dtypesFlag := flag.String("dtypes", "all", "float32|all|csv (all = every core.AllDTypes)")
	challengesFlag := flag.String("challenges", "all", "chase|flee|collect|teleport|shock|all|mock")
	camsFlag := flag.String("cams", "1-3", "camera count(s): 1|1,2,3|1-3|all (single→tricameral)")
	gridsFlag := flag.String("grids", "1-3", "mesh cube edge: 1|1,2,3|1x1x1,3x3x3|1-3|all")
	permute := flag.Bool("permute", false, "force full matrix (same as current defaults: layers=all dtypes=all …)")
	seed := flag.Int64("seed", 1, "rng seed")
	addr := flag.String("addr", "0.0.0.0:5151", "dash listen (0.0.0.0:5151 = all interfaces; empty = no dash)")
	game := flag.String("game", "", "empty/mock = Go challenges; ls20 = Python ARC-AGI-3 bridge")
	bridgePy := flag.String("bridge", "", "path to agent_bridge.py (default bridge/agent_bridge.py)")
	outJSON := flag.String("json", "", "legacy single-file dump (empty = skip; store always writes results.json)")
	ckpt := flag.String("ckpt", "test51_ckpt", "tide-like checkpoint dir (progress.json + history.json + results.json)")
	resume := flag.Bool("resume", true, "skip job IDs already in progress.json")
	autoStart := flag.Bool("autostart", false, "skip dash Start gate")
	flag.Parse()

	// Default go run . = full matrix. -permute forces the same (compat). Shrink with -layers/-dtypes.
	if *permute {
		*layersFlag = "all"
		*dtypesFlag = "all"
		*lrsFlag = "funny"
		*challengesFlag = "all"
		*camsFlag = "1-3"
		*gridsFlag = "1-3"
	}

	modes, err := parseModeList(*modesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	layers, err := parseLayerList(*layersFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	dtypes, err := parseDTypeList(*dtypesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	var lrs []float64
	if strings.TrimSpace(*lrsFlag) == "" {
		lrs = []float64{*lr}
	} else {
		lrs, err = parseLRList(*lrsFlag)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	challenges, err := parseChallengeList(*challengesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	cams, err := parseCamsList(*camsFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	grids, err := parseGridList(*gridsFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if strings.TrimSpace(*game) != "" && *game != "mock" {
		challenges = []string{"bridge"}
	}

	jobs := expandJobs(modes, layers, dtypes, lrs, challenges, cams, grids)
	trees := groupTrees(jobs)
	store := NewStore(*ckpt)
	prog, _, err := store.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "ckpt:", err)
		os.Exit(1)
	}
	done := doneSet(prog.DoneIDs)
	if !*resume {
		done = map[string]bool{}
		prog = &Progress{}
	}
	prog.Total = len(jobs)

	hub := newLiveHub()
	var results []modeResult
	if len(prog.Completed) > 0 && *resume {
		results = append(results, prog.Completed...)
	}
	disk, _ := store.LoadResults()
	if *resume && len(disk.Results) > len(results) {
		// results.json can be richer (phase rows); keep progress Completed as leaf source of truth for resume skip,
		// but prefer disk results when rebuilding reports if Completed is empty.
		if len(results) == 0 {
			results = append(results, disk.Results...)
		}
	}

	// Seed consolidation reports BEFORE Start so the dash shows them while waiting.
	if *resume {
		var seeded []treeReport
		if len(disk.Reports) > 0 {
			seeded = disk.Reports
		} else {
			seeded = rebuildReportsFromResults(trees, results, done)
		}
		if len(seeded) > 0 {
			enrichSrc := disk.Results
			if len(enrichSrc) == 0 {
				enrichSrc = results
			}
			if len(enrichSrc) > 0 && enrichReportsFromResults(seeded, enrichSrc) {
				fmt.Printf("📎 backfilled throughput on %d report(s) from ckpt results\n", len(seeded))
				bestID := prog.BestID
				if disk.BestID != "" {
					bestID = disk.BestID
				}
				saveRes := enrichSrc
				if len(disk.Results) > 0 {
					saveRes = disk.Results
				}
				_ = store.SaveResults(map[string]any{
					"results": saveRes,
					"reports": seeded,
					"best_id": bestID,
					"jobs":    len(jobs),
					"trees":   len(trees),
					"ckpt":    *ckpt,
				})
			}
			hub.seedReports(seeded)
			fmt.Printf("📂 restored %d consolidation report(s) from ckpt (visible before Start)\n", len(seeded))
		}
	}

	var dash *dashServer
	if strings.TrimSpace(*addr) != "" {
		listen := normalizeDashAddr(*addr)
		dash = newDashServer(listen, hub)
		go func() {
			if err := dash.listen(); err != nil {
				fmt.Fprintf(os.Stderr, "dash: %v\n", err)
			}
		}()
		port := dashPort(listen)
		fmt.Printf("dash listening on %s  →  http://<this-host-ip>:%s  (all interfaces, Start to begin)\n", listen, port)
		if !*autoStart {
			dash.awaitStart()
		} else {
			dash.signalStart()
		}
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  test51 — trees: mode×layer×dtype×challenge → LR↑×cams×grids           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	pending := 0
	for _, j := range jobs {
		if !done[j.ID] {
			pending++
		}
	}
	fmt.Printf("Trees: %d  ·  pending leaves: %d / %d  (board shows one tree at a time)\n\n", len(trees), pending, len(jobs))

	globalIdx := 0
	for ti, tree := range trees {
		// skip fully-done trees
		allDone := true
		for _, j := range tree.Jobs {
			if !done[j.ID] {
				allDone = false
				break
			}
		}
		if allDone {
			globalIdx += len(tree.Jobs)
			continue
		}

		hub.beginTree(tree, ti+1, len(trees))
		fmt.Printf("── tree %d/%d  %s · %s/%s · %s  (%d leaves: LR↑×cam×grid) ──\n",
			ti+1, len(trees), tree.Mode, tree.Layer, tree.DType, tree.Challenge, len(tree.Jobs))

		var treeLeaves []modeResult
		for _, job := range tree.Jobs {
			globalIdx++
			i := globalIdx - 1
			if done[job.ID] {
				// keep board slot marked done if we have a prior result
				for _, prev := range results {
					if prev.ID == job.ID && (prev.Phase == "train" || prev.Phase == "") {
						hub.finishLeaf(prev)
						treeLeaves = append(treeLeaves, prev)
						break
					}
				}
				continue
			}
			prog.NextIndex = i
			prog.Current = job.ID
			prog.Phase = "train"
			_ = store.SaveProgress(prog)
			hub.setMeta(job.ID, "train", i+1, len(jobs))
			hub.setStatus(fmt.Sprintf("tree %d/%d leaf · train %s", ti+1, len(trees), shortID(job.ID)))

			fmt.Printf("▶ [%d/%d] %s\n", i+1, len(jobs), job.ID)
			trainR := runCfg{
				mode: job.Mode, dur: *dur, win: *win, lr: job.LR, seed: *seed + int64(i)*17,
				thinkK: *thinkK, game: *game, challenge: job.Challenge, bridgePy: *bridgePy,
				layer: job.Layer, dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: true, phase: "train", jobID: job.ID, hub: hub,
			}.run()
			trainR.ID = job.ID
			trainR.Layer = job.Layer
			trainR.DType = job.DType.String()
			trainR.LR = job.LR
			trainR.Challenge = job.Challenge
			trainR.Cams = job.Cams
			trainR.GridN = job.GridN
			trainR.Phase = "train"
			_ = store.AppendHistory(HistoryPoint{
				At: time.Now().UTC(), JobID: job.ID, Phase: "train",
				Acc: trainR.Acc, Score: trainR.Score, Avail: trainR.Avail, Levels: trainR.Levels, LR: job.LR,
			})

			last := trainR
			if trainR.Err == "" && *afterFreeze > 0 && len(trainR.Weights) > 0 {
				prog.Phase = "after_freeze"
				hub.setMeta(job.ID, "after_freeze", i+1, len(jobs))
				hub.setStatus(fmt.Sprintf("tree %d/%d · freeze %s", ti+1, len(trees), shortID(job.ID)))
				fr := runCfg{
					mode: job.Mode, initW: trainR.Weights, dur: *afterFreeze, win: *win, lr: job.LR,
					seed: *seed + int64(i)*17 + 1, thinkK: *thinkK, game: *game, challenge: job.Challenge,
					bridgePy: *bridgePy, layer: job.Layer, dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: false,
					phase: "after_freeze", jobID: job.ID + "|freeze", hub: hub,
				}.run()
				fr.ID = job.ID + "|after_freeze"
				fr.Layer, fr.DType, fr.LR, fr.Challenge = job.Layer, job.DType.String(), job.LR, job.Challenge
				fr.Cams, fr.GridN = job.Cams, job.GridN
				fr.Phase = "after_freeze"
				fr.AccDelta = fr.Acc - trainR.Acc
				fr.ScoreDelta = fr.Score - trainR.Score
				_ = store.AppendHistory(HistoryPoint{
					At: time.Now().UTC(), JobID: job.ID, Phase: "after_freeze",
					Acc: fr.Acc, Score: fr.Score, Avail: fr.Avail, Levels: fr.Levels, LR: job.LR,
					Note: fmt.Sprintf("Δacc=%.1f Δscore=%.0f", fr.AccDelta, fr.ScoreDelta),
				})
				last = fr
				results = append(results, fr)
			}

			if trainR.Err == "" && *afterTrain > 0 && len(last.Weights) > 0 {
				prog.Phase = "after_train"
				hub.setMeta(job.ID, "after_train", i+1, len(jobs))
				hub.setStatus(fmt.Sprintf("tree %d/%d · after-train %s", ti+1, len(trees), shortID(job.ID)))
				baseW := last.Weights
				if len(trainR.Weights) > 0 {
					baseW = trainR.Weights
				}
				at := runCfg{
					mode: job.Mode, initW: baseW, dur: *afterTrain, win: *win, lr: job.LR,
					seed: *seed + int64(i)*17 + 2, thinkK: *thinkK, game: *game, challenge: job.Challenge,
					bridgePy: *bridgePy, layer: job.Layer, dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: true,
					phase: "after_train", jobID: job.ID + "|after", hub: hub,
				}.run()
				at.ID = job.ID + "|after_train"
				at.Layer, at.DType, at.LR, at.Challenge = job.Layer, job.DType.String(), job.LR, job.Challenge
				at.Cams, at.GridN = job.Cams, job.GridN
				at.Phase = "after_train"
				at.AccDelta = at.Acc - trainR.Acc
				at.ScoreDelta = at.Score - trainR.Score
				_ = store.AppendHistory(HistoryPoint{
					At: time.Now().UTC(), JobID: job.ID, Phase: "after_train",
					Acc: at.Acc, Score: at.Score, Avail: at.Avail, Levels: at.Levels, LR: job.LR,
					Note: fmt.Sprintf("Δacc=%.1f Δscore=%.0f (self-improve past stop)", at.AccDelta, at.ScoreDelta),
				})
				trainR.Weights = at.Weights
				trainR.AccDelta = at.AccDelta
				trainR.ScoreDelta = at.ScoreDelta
				if trainR.Acc != 0 {
					trainR.ImprovePct = 100 * at.AccDelta / trainR.Acc
					at.ImprovePct = trainR.ImprovePct
				}
				results = append(results, at)
				last = at
			}

			if trainR.Err != "" {
				fmt.Printf("  ERR %s\n", trainR.Err)
			} else {
				fmt.Printf("  Acc %.1f Score %.0f Avail %.1f Levels %d  (Δacc %+.1f / %+.1f%%)\n",
					trainR.Acc, trainR.Score, trainR.Avail, trainR.Levels, trainR.AccDelta, trainR.ImprovePct)
			}
			results = append(results, trainR)
			treeLeaves = append(treeLeaves, trainR)
			hub.finishLeaf(trainR)
			prog.DoneIDs = append(prog.DoneIDs, job.ID)
			done[job.ID] = true
			prog.Completed = append(prog.Completed, stripWeights(trainR))
			if trainR.Err == "" && (prog.BestID == "" || trainR.Score > prog.BestScore ||
				(trainR.Score == prog.BestScore && trainR.Acc > prog.BestAcc)) {
				prog.BestID = job.ID
				prog.BestScore = trainR.Score
				prog.BestAcc = trainR.Acc
			}
			prog.Phase = ""
			prog.Current = ""
			_ = store.SaveProgress(prog)
			_ = store.SaveResults(map[string]any{
				"results": results,
				"reports": hub.snapshot().Reports,
				"best_id": prog.BestID,
				"jobs":    len(jobs),
				"trees":   len(trees),
				"ckpt":    *ckpt,
			})
			hub.setLPDs(rebuildLPDByChallenge(results))
		}

		rep := summarizeTree(tree, treeLeaves)
		hub.finishTree(rep)
		fmt.Printf("✓ tree done  best Acc=%.1f Score=%.0f  %s (lr=%g cam=%d grid=%d)\n\n",
			rep.BestAcc, rep.BestScore, shortID(rep.BestID), rep.BestLR, rep.BestCams, rep.BestGrid)
		_ = store.SaveResults(map[string]any{
			"results": results,
			"reports": hub.snapshot().Reports,
			"best_id": prog.BestID,
			"jobs":    len(jobs),
			"trees":   len(trees),
			"ckpt":    *ckpt,
		})
	}

	byChal := rebuildLPDByChallenge(results)
	hub.setLPDs(byChal)
	printBoard(results, byChal)

	// Promote from the challenge that owns BestID (never cross-game LPD).
	lpd := lucy.LPD{}
	chalForPromote := ""
	for _, r := range results {
		if prog.BestID != "" && r.ID == prog.BestID {
			chalForPromote = r.Challenge
			break
		}
	}
	if chalForPromote != "" {
		lpd = byChal[chalForPromote]
	}
	if lpd.N == 0 {
		for _, c := range allChallenges() {
			if byChal[c].N > 0 {
				lpd = byChal[c]
				chalForPromote = c
				break
			}
		}
	}

	champMode := lpd.LiveChamp.Mode
	if champMode == "" {
		champMode = lpd.Champ.Mode
	}
	if champMode != "" && *promote > 0 {
		var champ *modeResult
		// Prefer the tide best job's train row (weights already include after_train).
		for i := range results {
			r := &results[i]
			if r.Err != "" || len(r.Weights) == 0 {
				continue
			}
			if prog.BestID != "" && r.ID == prog.BestID && (r.Phase == "train" || r.Phase == "") {
				champ = r
				break
			}
		}
		if champ == nil {
			for i := range results {
				r := &results[i]
				if r.Err != "" || len(r.Weights) == 0 {
					continue
				}
				if chalForPromote != "" && r.Challenge != chalForPromote {
					continue
				}
				if r.Phase != "" && r.Phase != "train" && r.Phase != "after_train" {
					continue
				}
				if r.Mode != champMode && !strings.Contains(r.ID, champMode) {
					continue
				}
				if champ == nil || r.Score > champ.Score {
					champ = r
				}
			}
		}
		if champ != nil {
			fmt.Printf("\nLPD promote [%s]: %s for %s (weight copy → keep thinking+training)\n",
				chalForPromote, champ.ID, *promote)
			pm, _ := parallel.ParseTrainMode(champ.Mode)
			dt := core.ParseDType(champ.DType)
			pr := runCfg{
				mode: pm, initW: champ.Weights, dur: *promote, win: *win, lr: champ.LR,
				seed: *seed + 999, thinkK: *thinkK, game: *game, challenge: champ.Challenge,
				bridgePy: *bridgePy, layer: champ.Layer, dtype: dt, cams: champ.Cams, gridN: champ.GridN, train: true,
				phase: "promote", jobID: "promote|" + champ.ID, hub: hub, promoted: true,
			}.run()
			pr.Promoted = true
			pr.ID = "promote|" + champ.ID
			pr.Phase = "promote"
			results = append(results, pr)
			hub.setStatus(fmt.Sprintf("promoted Acc=%.1f Score=%.0f", pr.Acc, pr.Score))
		}
	}

	_ = store.SaveResults(map[string]any{
		"results":          results,
		"reports":          hub.snapshot().Reports,
		"lpd_by_challenge": byChal,
		"best_id":          prog.BestID,
		"jobs":             len(jobs),
		"trees":            len(trees),
		"ckpt":             *ckpt,
	})
	fmt.Printf("💾 %s/{progress,history,results}.json\n", *ckpt)

	if strings.TrimSpace(*outJSON) != "" {
		mustWrite(*outJSON, map[string]any{"results": results, "lpd_by_challenge": byChal})
		fmt.Printf("💾 %s\n", *outJSON)
	}

	if dash != nil {
		hub.setStatus("sweep complete — dash still live")
		fmt.Println("dash still serving; Ctrl-C to exit")
		select {}
	}
}

func (c runCfg) run() modeResult {
	return runModeInner(c)
}

func stripWeights(r modeResult) modeResult {
	r.Weights = nil
	r.Lucy.Windows = nil
	return r
}

func normalizeDashAddr(spec string) string {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return ""
	}
	// ":5151" or bare port → bind all interfaces (remote LAN access).
	if strings.HasPrefix(spec, ":") {
		return "0.0.0.0" + spec
	}
	host, port, err := net.SplitHostPort(spec)
	if err != nil {
		// bare "5151"
		if _, e := strconv.Atoi(spec); e == nil {
			return "0.0.0.0:" + spec
		}
		return spec
	}
	if host == "" || host == "localhost" {
		return "0.0.0.0:" + port
	}
	return spec
}

func dashPort(listen string) string {
	if listen == "" {
		return "5151"
	}
	if i := strings.LastIndex(listen, ":"); i >= 0 && i < len(listen)-1 {
		return listen[i+1:]
	}
	return listen
}

func shortID(id string) string {
	if len(id) <= 64 {
		return id
	}
	return id[:61] + "…"
}

func nz(s, d string) string {
	if strings.TrimSpace(s) == "" {
		return d
	}
	return s
}

func gameLabel(game string) string {
	g := strings.TrimSpace(game)
	if g == "" || g == "mock" {
		return "mock challenges (Go)"
	}
	return g + " (bridge)"
}

func parseModeList(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return parallel.AllNamedTrainModes(), nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		key := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(tok, "_", ""), "-", ""))
		var m parallel.TrainMode
		var err error
		switch key {
		case "stepsgd", "stepbp":
			m = parallel.ModeStepBP
		default:
			m, err = parallel.ParseTrainMode(tok)
			if err != nil {
				return nil, err
			}
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

func runMode(
	mode parallel.TrainMode,
	dur, win time.Duration,
	lr float64,
	seed int64,
	thinkK int,
	game, bridgePy string,
	hub *liveHub,
) modeResult {
	return runCfg{
		mode: mode, dur: dur, win: win, lr: lr, seed: seed, thinkK: thinkK,
		game: game, challenge: chalChase, bridgePy: bridgePy,
		layer: "dense", dtype: core.DTypeFloat32, cams: 1, gridN: 1, train: true, phase: "train", hub: hub,
	}.run()
}

func runModeInner(c runCfg) modeResult {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	label := c.mode.String()
	if c.jobID != "" {
		label = c.jobID
	}
	r := modeResult{
		Mode: c.mode.String(), ThinkK: c.thinkK, Promoted: c.promoted,
		Layer: c.layer, DType: c.dtype.String(), LR: c.lr, Challenge: c.challenge,
		Cams: c.cams, GridN: c.gridN,
		Phase: c.phase, ID: c.jobID,
	}
	rng := rand.New(rand.NewSource(c.seed))
	st, err := buildPolicyNetEx(rng, core.BackendCPUTiled, c.layer, c.dtype, c.cams)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	if c.initW != nil {
		if err := restoreWeights(st, c.initW); err != nil {
			r.Err = "restore: " + err.Error()
			return r
		}
	}
	r.Lucy.WeightBytes = stackWeightBytes(st)
	r.RAMKiB = float64(r.Lucy.WeightBytes) / 1024

	var grid *architecture.Grid
	if c.mode.RequiresGrid() {
		g, gerr := placeForMesh(st, c.gridN)
		if gerr != nil {
			r.Err = "place: " + gerr.Error()
			return r
		}
		grid = g
	}

	env, err := openChallengeOrGame(c.game, c.challenge, c.seed, c.bridgePy)
	if err != nil {
		r.Err = "env: " + err.Error()
		return r
	}
	defer env.Close()

	fr, err := env.Reset()
	if err != nil {
		r.Err = "reset: " + err.Error()
		return r
	}
	if c.hub != nil {
		c.hub.setFrame(fr, Action{}, label, 0)
	}

	nWin := int(c.dur / c.win)
	if nWin < 1 {
		nWin = 1
	}
	wins := make([]lucy.Window, nWin)
	var infSum, trSum time.Duration
	start := time.Now()
	maxLevels := 0

	for time.Since(start) < c.dur {
		elapsed := time.Since(start)
		wi := int(elapsed / c.win)
		if wi >= nWin {
			wi = nWin - 1
		}
		wins[wi].Phase = c.mode.Short()

		if fr.State == stateWin || fr.State == stateGameOver {
			fr, err = env.Reset()
			if err != nil {
				break
			}
		}

		tInf := startWork()
		tr, terr := thinkThenAct(st, grid, c.mode, fr, c.thinkK)
		inf := tInf.elapsed()
		infSum += inf
		if terr != nil || tr.Post == nil {
			continue
		}

		oracle := env.OracleAction(fr)
		pred := tr.Action.ID
		hard := 0.0
		if oracle >= 0 && pred == oracle {
			hard = 100
		}
		lab := oracle
		if lab < 0 {
			lab = pred
		}
		actionLogits := tr.Post.Data
		if len(actionLogits) > nActions {
			actionLogits = actionLogits[:nActions]
		}
		soft := lucy.SoftAccProb(softmaxPTrue(actionLogits, lab), 1)

		gx, gy := goalCoordsNorm(fr)
		y := targetFromOracle(lab, fr, gx, gy)

		r.Lucy.TotalOutputs++
		w := &wins[wi]
		w.Outputs++
		w.InferMs += inf.Seconds() * 1000
		w.Accuracy += hard
		w.SoftAcc += soft
		if hard == 100 {
			w.Correct++
			r.Lucy.TotalCorrect++
		}

		if c.train {
			r.Lucy.TotalTrain++
			w.TrainSteps++
			tTr := startWork()
			trainSample(st, grid, tr.MeshFwd, tr.Tape, tr.X, y, c.mode, c.lr)
			trMs := tTr.elapsed()
			trSum += trMs
			w.TrainMs += trMs.Seconds() * 1000
		}

		next, serr := env.Step(tr.Action)
		if serr != nil {
			r.Err = "step: " + serr.Error()
			break
		}
		if next.LevelsCompleted > maxLevels {
			maxLevels = next.LevelsCompleted
		}
		fr = next
		if c.hub != nil {
			c.hub.setFrame(fr, tr.Action, label, tr.ThinkK)
			secs := math.Max(time.Since(start).Seconds(), 1e-6)
			c.hub.pulseMode(modeResult{
				ID: label, Mode: c.mode.String(), Layer: c.layer, DType: c.dtype.String(),
				LR: c.lr, Challenge: c.challenge, Phase: c.phase,
				Acc: runningAcc(wins), Soft: soft,
				Avail: availOf(infSum, trSum), Thru: float64(r.Lucy.TotalOutputs) / secs,
				RAMKiB: r.RAMKiB, Levels: maxLevels, Actions: r.Lucy.TotalOutputs, ThinkK: c.thinkK,
			})
		}
	}

	for i := range wins {
		if wins[i].Outputs > 0 {
			wins[i].Accuracy /= float64(wins[i].Outputs)
			wins[i].SoftAcc /= float64(wins[i].Outputs)
		}
	}
	r.Lucy.Windows = wins
	r.Lucy.Duration = time.Since(start)
	r.Lucy.InferMs = infSum.Seconds() * 1000
	r.Lucy.TrainMs = trSum.Seconds() * 1000
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: 1, ConsThreshold: lucy.ConsThreshold})
	r.Acc = r.Lucy.AvgAccuracy
	r.Soft = r.Lucy.SoftAcc
	r.Avail = r.Lucy.Availability
	r.Thru = r.Lucy.Throughput
	r.Score = r.Lucy.Score
	r.Levels = maxLevels
	r.Actions = r.Lucy.TotalOutputs
	r.Weights = weightSnapshot(st)
	if c.hub != nil {
		c.hub.finishMode(r)
	}
	return r
}

func runningAcc(wins []lucy.Window) float64 {
	var o, c float64
	for _, w := range wins {
		o += float64(w.Outputs)
		c += float64(w.Correct)
	}
	if o == 0 {
		return 0
	}
	return 100 * c / o
}

func availOf(inf, tr time.Duration) float64 {
	t := inf + tr
	if t <= 0 {
		return 100 // freeze phase: all infer
	}
	return 100 * float64(inf) / float64(t)
}


func lpdSample(r modeResult) (lucy.Sample, bool) {
	phaseOK := r.Phase == "" || r.Phase == "train" || r.Phase == "after_train"
	if r.Err != "" || !phaseOK {
		return lucy.Sample{}, false
	}
	id := r.Mode
	if r.ID != "" {
		id = r.ID
	}
	arch := nz(r.Layer, "dense-think")
	if r.Cams > 1 {
		arch = fmt.Sprintf("%s|%s|%s", arch, camName(r.Cams), gridName(r.GridN))
	} else if r.GridN > 1 {
		arch = fmt.Sprintf("%s|%s", arch, gridName(r.GridN))
	}
	return lucy.Sample{
		ID: id, Mode: r.Mode, DType: nz(r.DType, "float32"), Arch: arch,
		Score: r.Score, Soft: r.Soft, Acc: r.Acc, Thru: r.Thru, Avail: r.Avail, RAMKiB: r.RAMKiB,
	}, true
}

// rebuildLPDByChallenge builds a separate LPD board per challenge so chase Acc
// never ranks against teleport Acc (apples-to-oranges).
func rebuildLPDByChallenge(rows []modeResult) map[string]lucy.LPD {
	buckets := map[string][]lucy.Sample{}
	for _, r := range rows {
		s, ok := lpdSample(r)
		if !ok {
			continue
		}
		chal := r.Challenge
		if chal == "" {
			chal = "unknown"
		}
		buckets[chal] = append(buckets[chal], s)
	}
	out := make(map[string]lucy.LPD, len(buckets))
	for chal, pts := range buckets {
		out[chal] = lucy.BuildLPD(pts)
	}
	return out
}

func orderedLPDChallenges(by map[string]lucy.LPD) []string {
	var out []string
	seen := map[string]bool{}
	for _, c := range allChallenges() {
		if _, ok := by[c]; ok {
			out = append(out, c)
			seen[c] = true
		}
	}
	var extra []string
	for c := range by {
		if !seen[c] {
			extra = append(extra, c)
		}
	}
	sort.Strings(extra)
	return append(out, extra...)
}

func printBoard(rows []modeResult, byChal map[string]lucy.LPD) {
	for _, chal := range orderedLPDChallenges(byChal) {
		fmt.Printf("\n── LPD · %s ──\n", chal)
		fmt.Println("╔══════════════════════╦═══════╦═══════╦═══════╦════════╦════════╗")
		fmt.Println("║ Job / Mode           ║ Acc   ║ Soft  ║ Avail ║ Score  ║ Lv ║ Δacc ║")
		fmt.Println("╠══════════════════════╬═══════╬═══════╬═══════╬════════╬════════╣")
		n := 0
		for _, r := range rows {
			if r.Challenge != chal {
				continue
			}
			n++
			name := r.Mode
			if r.ID != "" {
				name = shortID(r.ID)
			}
			if r.Err != "" {
				fmt.Printf("║ %-20s ║ ERR %s\n", clip(name, 20), r.Err)
				continue
			}
			fmt.Printf("║ %-20s ║ %5.1f ║ %5.1f ║ %5.1f ║ %6.0f ║ %2d ║ %+5.1f ║\n",
				clip(name, 20), r.Acc, r.Soft, r.Avail, r.Score, r.Levels, r.AccDelta)
		}
		if n == 0 {
			fmt.Println("║ (no rows)            ║")
		}
		fmt.Println("╚══════════════════════╩═══════╩═══════╩═══════╩════════╩════════╝")
		lpd := byChal[chal]
		fmt.Printf("LPD champ=%s live=%s gold-std=%s (n=%d)\n",
			lpd.Champ.Mode, lpd.LiveChamp.Mode, lpd.GoldStd.Mode, lpd.N)
	}
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
