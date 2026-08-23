package main

import (
	"math/rand"
	"testing"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

func TestMockEnvOracleAndDims(t *testing.T) {
	env := NewMockEnv(1, "mock-test")
	fr, err := env.Reset()
	if err != nil {
		t.Fatal(err)
	}
	if len(fr.Layers) != 1 || len(fr.Layers[0]) != frameSize || len(fr.Layers[0][0]) != frameSize {
		t.Fatalf("frame shape got %d want %d", len(fr.Layers[0]), frameSize)
	}
	obs := encodeObs(fr)
	if len(obs) != obsDim {
		t.Fatalf("obs dim %d want %d", len(obs), obsDim)
	}
	oracle := env.OracleAction(fr)
	if oracle < 1 || oracle > 7 {
		t.Fatalf("oracle %d out of range", oracle)
	}
	fr2, err := env.Step(Action{ID: oracle})
	if err != nil {
		t.Fatal(err)
	}
	if fr2.State == "" {
		t.Fatal("empty state")
	}
	_ = env.Close()
}

func TestThinkDimsAndLoop(t *testing.T) {
	if inDim != obsDim+thinkDim {
		t.Fatalf("inDim %d != obsDim+thinkDim %d", inDim, obsDim+thinkDim)
	}
	if outDim != nActions+thinkDim+coordDim {
		t.Fatalf("outDim %d mismatch", outDim)
	}
	rng := rand.New(rand.NewSource(2))
	st, err := buildPolicyNet(rng, core.BackendCPUTiled)
	if err != nil {
		t.Fatal(err)
	}
	env := NewMockEnv(2, "t")
	fr := envMustReset(env)
	tr, err := thinkThenAct(st, nil, parallel.ModeNormalBP, fr, 3)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Post == nil || len(tr.Post.Data) < outDim {
		t.Fatalf("post len %d", len(tr.Post.Data))
	}
	if tr.ThinkK != 3 {
		t.Fatalf("thinkK %d", tr.ThinkK)
	}
	if tr.Action.ID < 0 || tr.Action.ID >= nActions {
		t.Fatalf("action %d", tr.Action.ID)
	}
}

func TestParseModesAll(t *testing.T) {
	ms, err := parseModeList("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) < 20 {
		t.Fatalf("expected many named modes, got %d", len(ms))
	}
	ms2, err := parseModeList("sgd,step_sgd,TweenSplitSparse")
	if err != nil {
		t.Fatal(err)
	}
	if len(ms2) != 3 {
		t.Fatalf("got %d modes: %v", len(ms2), ms2)
	}
}

func TestExpandFunnyJobs(t *testing.T) {
	lrs, err := parseLRList("funny")
	if err != nil {
		t.Fatal(err)
	}
	if len(lrs) != 8 || lrs[0] != 0.02 || lrs[len(lrs)-1] != 1_000_000 {
		t.Fatalf("funny lrs: %v", lrs)
	}
	chs, err := parseChallengeList("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(chs) != 5 {
		t.Fatalf("challenges %d", len(chs))
	}
	layers, _ := parseLayerList("all")
	dtypes, _ := parseDTypeList("float32")
	modes, _ := parseModeList("NormalBP")
	jobs := expandJobs(modes, layers, dtypes, lrs, chs, []int{1}, []int{1})
	want := 1 * 4 * 1 * 8 * 5 * 1 * 1
	if len(jobs) != want {
		t.Fatalf("jobs %d want %d", len(jobs), want)
	}
	// LR must climb: first job lr <= last job lr
	if jobs[0].LR > jobs[len(jobs)-1].LR {
		t.Fatalf("LR order not ascending: first=%g last=%g", jobs[0].LR, jobs[len(jobs)-1].LR)
	}
}

func TestExpandCamsGrids(t *testing.T) {
	modes, _ := parseModeList("NormalBP")
	layers, _ := parseLayerList("dense")
	dtypes, _ := parseDTypeList("float32")
	lrs := []float64{0.02, 2}
	chs := []string{chalChase}
	jobs := expandJobs(modes, layers, dtypes, lrs, chs, []int{1, 2, 3}, []int{1, 2, 3})
	want := 1 * 1 * 1 * 2 * 1 * 3 * 3
	if len(jobs) != want {
		t.Fatalf("jobs %d want %d", len(jobs), want)
	}
	// within a tree, LR climbs: all lr=0.02 before any lr=2
	seenHigh := false
	for _, j := range jobs {
		if j.LR > 0.02 {
			seenHigh = true
		}
		if seenHigh && j.LR == 0.02 {
			t.Fatalf("low LR appeared after high LR: %+v", j)
		}
	}
}


func TestChallengeFleeOracle(t *testing.T) {
	env := newChallengeEnv(chalFlee, 7)
	fr, err := env.Reset()
	if err != nil {
		t.Fatal(err)
	}
	a := env.OracleAction(fr)
	if a < 1 || a > 7 {
		t.Fatalf("oracle %d", a)
	}
	_, err = env.Step(Action{ID: a})
	if err != nil {
		t.Fatal(err)
	}
}

func TestStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	p := &Progress{Total: 3, NextIndex: 1, DoneIDs: []string{"a"}, BestID: "a", BestScore: 1}
	if err := s.SaveProgress(p); err != nil {
		t.Fatal(err)
	}
	if err := s.AppendHistory(HistoryPoint{JobID: "a", Phase: "train", Acc: 10}); err != nil {
		t.Fatal(err)
	}
	p2, hist, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if p2.BestID != "a" || len(hist) != 1 {
		t.Fatalf("got %+v hist=%d", p2, len(hist))
	}
}

func TestShortRaceNormalBP(t *testing.T) {
	hub := newLiveHub()
	hub.signalStart()
	r := runMode(parallel.ModeNormalBP, 400*time.Millisecond, 100*time.Millisecond, 0.05, 3, 2, "", "", hub)
	if r.Err != "" {
		t.Fatal(r.Err)
	}
	if r.Actions < 1 {
		t.Fatalf("no actions: %+v", r)
	}
	if r.Lucy.WeightBytes <= 0 {
		t.Fatal("missing weight bytes")
	}
}

func TestAfterFreezeNoTrain(t *testing.T) {
	hub := newLiveHub()
	train := runCfg{
		mode: parallel.ModeNormalBP, dur: 300 * time.Millisecond, win: 100 * time.Millisecond,
		lr: 0.05, seed: 9, thinkK: 2, challenge: chalChase, layer: "dense",
		dtype: core.DTypeFloat32, train: true, phase: "train", jobID: "t", hub: hub,
	}.run()
	if train.Err != "" || len(train.Weights) == 0 {
		t.Fatalf("train: %+v", train)
	}
	fr := runCfg{
		mode: parallel.ModeNormalBP, initW: train.Weights, dur: 200 * time.Millisecond,
		win: 100 * time.Millisecond, lr: 0.05, seed: 10, thinkK: 2, challenge: chalChase,
		layer: "dense", dtype: core.DTypeFloat32, train: false, phase: "after_freeze",
		jobID: "f", hub: hub,
	}.run()
	if fr.Err != "" {
		t.Fatal(fr.Err)
	}
	if fr.Lucy.TotalTrain != 0 {
		t.Fatalf("freeze should not train, got %d", fr.Lucy.TotalTrain)
	}
	if fr.Avail < 99 {
		t.Fatalf("freeze avail %.1f", fr.Avail)
	}
}


func TestGroupTrees(t *testing.T) {
	modes, _ := parseModeList("NormalBP,StepBP")
	layers, _ := parseLayerList("dense")
	dtypes, _ := parseDTypeList("float32")
	lrs := []float64{0.02, 2}
	chs := []string{chalChase, chalFlee}
	jobs := expandJobs(modes, layers, dtypes, lrs, chs, []int{1, 2}, []int{1})
	trees := groupTrees(jobs)
	// 2 modes × 1 layer × 1 dtype × 2 challenges = 4 trees
	if len(trees) != 4 {
		t.Fatalf("trees %d want 4", len(trees))
	}
	for _, tr := range trees {
		if len(tr.Jobs) != 4 { // 2 LR × 2 cams × 1 grid
			t.Fatalf("tree %s leaves %d want 4", tr.Key, len(tr.Jobs))
		}
		if tr.Jobs[0].LR > tr.Jobs[len(tr.Jobs)-1].LR {
			t.Fatalf("tree %s LR not ascending", tr.Key)
		}
	}
	// first tree should be first mode + first challenge; finish before next mode's leaves
	if trees[0].Mode == trees[len(trees)-1].Mode && trees[0].Challenge == trees[len(trees)-1].Challenge {
		t.Fatal("expected distinct trees")
	}
}

func TestTreeReportPDFAndHub(t *testing.T) {
	leaves := []modeResult{
		{ID: "a", Mode: "NormalBP", Layer: "dense", DType: "float32", Challenge: chalChase,
			LR: 0.02, Cams: 1, GridN: 1, Phase: "after_train", Acc: 30, Soft: 40, Score: 300, AccDelta: 2},
		{ID: "b", Mode: "NormalBP", Layer: "dense", DType: "float32", Challenge: chalChase,
			LR: 2, Cams: 2, GridN: 2, Phase: "after_train", Acc: 40, Soft: 50, Score: 400, AccDelta: 5},
	}
	tr := Tree{Key: "NormalBP|dense|float32|chase", Mode: "NormalBP", Layer: "dense", DType: "float32", Challenge: chalChase, Jobs: make([]Job, 2)}
	rep := summarizeTree(tr, leaves)
	if rep.BestID != "b" || len(rep.Rows) != 2 {
		t.Fatalf("best=%s rows=%d", rep.BestID, len(rep.Rows))
	}
	hub := newLiveHub()
	hub.finishTree(rep)
	got, ok := hub.reportByIndex(1)
	if !ok || got.URL != "/report/1" || got.PDF != "/report/1.pdf" {
		t.Fatalf("hub report %#v ok=%v", got, ok)
	}
	pdf := buildReportPDF(got)
	if len(pdf) < 200 || string(pdf[:4]) != "%PDF" {
		t.Fatalf("bad pdf len=%d head=%q", len(pdf), pdf[:min(8, len(pdf))])
	}
}
