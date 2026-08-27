package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/tide/report"
)

// cacheCell is one ok row from a test53 results.json (raw eval Acc only).
type cacheCell struct {
	Farm     string  `json:"farm"`
	Path     string  `json:"path,omitempty"`
	ID       string  `json:"id"`
	Layer    string  `json:"layer"`
	Mode     string  `json:"mode"`
	DType    string  `json:"dtype"`
	LR       float64 `json:"lr"`
	LRLabel  string  `json:"lr_label"`
	Acc      float64 `json:"acc"`       // top-level results.acc (evalRoute)
	TrainAcc float64 `json:"train_acc"` // lucy.avg_accuracy (teacher-forced)
	Soft     float64 `json:"soft"`
	Avail    float64 `json:"avail"`
	Thru     float64 `json:"thru"`
	Score    float64 `json:"score"`
	RAMKiB   float64 `json:"ram_kib"`
	Fit      float64 `json:"fit"` // Acc × Avail / 100 — serve+train proxy
}

type cacheFarm struct {
	Name    string `json:"name"`
	Rel     string `json:"rel"`
	Path    string `json:"path"`
	N       int    `json:"n"`
	BestAcc float64 `json:"best_acc"`
	BestID  string  `json:"best_id"`
}

type accAgg struct {
	Key       string  `json:"key"`
	N         int     `json:"n"`
	BestAcc   float64 `json:"best_acc"`
	MeanAcc   float64 `json:"mean_acc"`
	BestFit   float64 `json:"best_fit"`
	MeanFit   float64 `json:"mean_fit"`
	MeanAvail float64 `json:"mean_avail"`
	BestID    string  `json:"best_id"`
	BestFarm  string  `json:"best_farm"`
	BestLayer string  `json:"best_layer,omitempty"`
	BestMode  string  `json:"best_mode,omitempty"`
	BestDtype string  `json:"best_dtype,omitempty"`
}

// dtypeLayerBlock is dtype ranking inside one layer (or layer ranking inside one dtype).
type dtypeLayerBlock struct {
	Key  string   `json:"key"` // layer name or dtype name
	N    int      `json:"n"`
	Best float64  `json:"best_acc"`
	Rows []accAgg `json:"rows"`
}

type accSeriesPoint struct {
	LR        float64 `json:"lr"`
	LRLabel   string  `json:"lr_label"`
	N         int     `json:"n"`
	BestAcc   float64 `json:"best_acc"`
	MeanAcc   float64 `json:"mean_acc"`
	BestFit   float64 `json:"best_fit"`
	MeanAvail float64 `json:"mean_avail"`
}

type accSeries struct {
	Name   string           `json:"name"`
	Kind   string           `json:"kind"` // farm|mode|layer
	Points []accSeriesPoint `json:"points"`
}

// AccBoard is the Acc-first cache dashboard payload.
type AccBoard struct {
	Generated     time.Time         `json:"generated"`
	CacheRoot     string            `json:"cache_root"`
	NCells        int               `json:"n_cells"`
	NFarms        int               `json:"n_farms"`
	Farms         []cacheFarm       `json:"farms"`
	TopAcc        []cacheCell       `json:"top_acc"`
	TopFit        []cacheCell       `json:"top_fit"`
	Scatter       []cacheCell       `json:"scatter"`
	ByMode        []accAgg          `json:"by_mode"`
	ByLayer       []accAgg          `json:"by_layer"`
	ByDtype       []accAgg          `json:"by_dtype"` // layers combined — best dtype overall (+ which layer)
	ByFarm        []accAgg          `json:"by_farm"`
	DtypeByLayer  []dtypeLayerBlock `json:"dtype_by_layer"`  // per layer: dtype Acc rank
	LayerByDtype  []dtypeLayerBlock `json:"layer_by_dtype"`  // per dtype: layer Acc rank
	FarmSeries    []accSeries       `json:"farm_series"`
	ModeSeries    []accSeries       `json:"mode_series"`
	LayerSeries   []accSeries       `json:"layer_series"`
	Note          string            `json:"note"`
}

type cacheIndex struct {
	mu        sync.RWMutex
	root      string
	loadedAt  time.Time
	cells     []cacheCell
	farms     []cacheFarm
	board     AccBoard
	tides     []report.NamedTideReport
}

func newCacheIndex(root string) *cacheIndex {
	return &cacheIndex{root: strings.TrimSpace(root)}
}

func (c *cacheIndex) reload() error {
	if c == nil || c.root == "" {
		return fmt.Errorf("empty cache root")
	}
	root, err := filepath.Abs(c.root)
	if err != nil {
		return err
	}
	var farms []cacheFarm
	var cells []cacheCell
	var tides []report.NamedTideReport

	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == ".git" || base == "node_modules" || strings.HasPrefix(base, ".") {
				if path != root {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if d.Name() != "results.json" {
			return nil
		}
		dir := filepath.Dir(path)
		rel, _ := filepath.Rel(root, dir)
		if rel == "." {
			rel = filepath.Base(dir)
		}
		farmName := farmDisplayName(rel)
		loaded, nOk, bestAcc, bestID, err := loadResultsFile(path, farmName)
		if err != nil || nOk == 0 {
			return nil
		}
		farms = append(farms, cacheFarm{
			Name: farmName, Rel: rel, Path: dir, N: nOk, BestAcc: bestAcc, BestID: bestID,
		})
		cells = append(cells, loaded...)
		pts := make([]report.CellPoint, 0, len(loaded))
		for _, row := range loaded {
			pts = append(pts, report.CellPoint{
				Tide: farmName, Layer: row.Layer, ID: row.ID, Mode: row.Mode, DType: row.DType,
				Arch: row.Layer, Score: row.Score, Soft: row.Soft, Acc: row.Acc,
				Avail: row.Avail, Thru: row.Thru, Adapt: 0, RAMKiB: row.RAMKiB,
			})
		}
		// Dotted name so report.ParseMachineFromPeer keeps the full farm id.
		tides = append(tides, report.NamedTideReport{
			Name: farmName,
			Report: report.TideReport{
				Kind: "cache", ID: farmName, Task: "test53-dayroute",
				Subtitle: rel, Recorded: nOk, Cells: pts,
			},
		})
		return nil
	})
	if err != nil {
		return err
	}
	sort.Slice(farms, func(i, j int) bool {
		if farms[i].BestAcc != farms[j].BestAcc {
			return farms[i].BestAcc > farms[j].BestAcc
		}
		return farms[i].Name < farms[j].Name
	})
	board := buildAccBoard(root, farms, cells)
	c.mu.Lock()
	c.root = root
	c.loadedAt = time.Now()
	c.cells = cells
	c.farms = farms
	c.board = board
	c.tides = tides
	c.mu.Unlock()
	return nil
}

func farmDisplayName(rel string) string {
	rel = strings.Trim(rel, `/`)
	rel = strings.ReplaceAll(rel, `\`, `/`)
	// Keep full path distinct for compare (dots, not _/- which ParseMachineFromPeer splits on).
	rel = strings.ReplaceAll(rel, "/", ".")
	rel = strings.ReplaceAll(rel, "_", ".")
	rel = strings.ReplaceAll(rel, "-", ".")
	return rel
}

type slimLucy struct {
	AvgAccuracy float64 `json:"avg_accuracy"`
}

type slimResult struct {
	ID     string   `json:"id"`
	Layer  string   `json:"layer"`
	Mode   string   `json:"mode"`
	DType  string   `json:"dtype"`
	LR     float64  `json:"lr"`
	Acc    float64  `json:"acc"`
	Soft   float64  `json:"soft"`
	Avail  float64  `json:"avail"`
	Thru   float64  `json:"thru"`
	Score  float64  `json:"score"`
	RAMKiB float64  `json:"ram_kib"`
	Err    string   `json:"err"`
	Lucy   slimLucy `json:"lucy"`
}

func loadResultsFile(path, farm string) (rows []cacheCell, nOk int, bestAcc float64, bestID string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, "", err
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	// Expect {"results":[...], ...}
	tok, err := dec.Token()
	if err != nil {
		return nil, 0, 0, "", err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, 0, 0, "", fmt.Errorf("%s: not an object", path)
	}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, 0, 0, "", err
		}
		key, _ := keyTok.(string)
		if key != "results" {
			var skip any
			if err := dec.Decode(&skip); err != nil {
				return nil, 0, 0, "", err
			}
			continue
		}
		tok, err := dec.Token()
		if err != nil {
			return nil, 0, 0, "", err
		}
		if delim, ok := tok.(json.Delim); !ok || delim != '[' {
			return nil, 0, 0, "", fmt.Errorf("%s: results not array", path)
		}
		for dec.More() {
			var r slimResult
			if err := dec.Decode(&r); err != nil {
				return nil, 0, 0, "", err
			}
			if strings.TrimSpace(r.Err) != "" {
				continue
			}
			if r.Acc <= 0 && r.Score <= 0 {
				continue
			}
			lr := r.LR
			lbl := ""
			if v, l, ok := report.ParseLRFromCellID(r.ID); ok {
				lr, lbl = v, l
			} else if lr != 0 {
				lbl = report.FormatLR(lr)
			}
			layer := r.Layer
			mode := r.Mode
			dtype := r.DType
			if layer == "" || mode == "" || dtype == "" {
				parts := strings.Split(r.ID, "|")
				if len(parts) >= 3 {
					if layer == "" {
						layer = parts[0]
					}
					if dtype == "" {
						dtype = parts[1]
					}
					if mode == "" {
						mode = parts[2]
					}
				}
			}
			cell := cacheCell{
				Farm: farm, Path: path, ID: r.ID, Layer: layer, Mode: shortMode(mode), DType: dtype,
				LR: lr, LRLabel: lbl, Acc: r.Acc, TrainAcc: r.Lucy.AvgAccuracy,
				Soft: r.Soft, Avail: r.Avail, Thru: r.Thru, Score: r.Score, RAMKiB: r.RAMKiB,
				Fit: r.Acc * r.Avail / 100,
			}
			rows = append(rows, cell)
			nOk++
			if cell.Acc > bestAcc || (cell.Acc == bestAcc && bestID == "") {
				bestAcc, bestID = cell.Acc, cell.ID
			}
		}
		// drain rest of object
		_, _ = dec.Token() // ]
		for dec.More() {
			_, _ = dec.Token()
			var skip any
			_ = dec.Decode(&skip)
		}
		_, _ = dec.Token() // }
		return rows, nOk, bestAcc, bestID, nil
	}
	return nil, 0, 0, "", io.ErrUnexpectedEOF
}

func shortMode(m string) string {
	repl := []struct{ old, neu string }{
		{"MeshTweenSplitSparse", "Mesh[T][S]Sparse"},
		{"StepTweenSplitSparse", "Step[T][S]Sparse"},
		{"TweenSplitSparse", "[T][S]Sparse"},
		{"MeshTweenSplitFastProxy", "Mesh[T][S][FP]"},
		{"StepTweenSplitFastProxy", "Step[T][S][FP]"},
		{"TweenSplitFastProxy", "[T][S][FP]"},
		{"MeshTweenSplit", "Mesh[T][S]"},
		{"StepTweenSplit", "Step[T][S]"},
		{"TweenSplitHeadProxyAsync", "[T][S][HP]Async"},
		{"StepTweenSplitHeadProxyAsync", "Step[T][S][HP]Async"},
		{"TweenSplitHeadProxy", "[T][S][HP]"},
		{"StepTweenSplitHeadProxy", "Step[T][S][HP]"},
		{"TweenSplitLinearCache", "[T][S][L]Cache"},
		{"StepTweenSplitLinearCache", "Step[T][S][L]Cache"},
		{"TweenSplitLinear", "[T][S][L]"},
		{"StepTweenSplitLinear", "Step[T][S][L]"},
		{"TweenSplit", "[T][S]"},
		{"StepTweenChain", "Step[T]Chain"},
		{"TweenChain", "[T]Chain"},
		{"MeshTween", "Mesh[T]"},
		{"StepTween", "Step[T]"},
		{"NormalBP", "sgd"},
		{"StepBP", "step_sgd"},
		{"MeshBP", "MeshBP"},
	}
	for _, r := range repl {
		if m == r.old {
			return r.neu
		}
	}
	return m
}

func buildAccBoard(root string, farms []cacheFarm, cells []cacheCell) AccBoard {
	out := AccBoard{
		Generated: time.Now(),
		CacheRoot: root,
		NCells:    len(cells),
		NFarms:    len(farms),
		Farms:     farms,
		Note:      "Acc = top-level results.acc (closed-loop eval). Fit = Acc×Avail/100 (serve+train). Not LPD.",
	}
	if len(cells) == 0 {
		return out
	}

	byAcc := append([]cacheCell(nil), cells...)
	sort.SliceStable(byAcc, func(i, j int) bool {
		if byAcc[i].Acc != byAcc[j].Acc {
			return byAcc[i].Acc > byAcc[j].Acc
		}
		if byAcc[i].Avail != byAcc[j].Avail {
			return byAcc[i].Avail > byAcc[j].Avail
		}
		return byAcc[i].Score > byAcc[j].Score
	})
	out.TopAcc = trimCells(byAcc, 250)

	byFit := append([]cacheCell(nil), cells...)
	sort.SliceStable(byFit, func(i, j int) bool {
		if byFit[i].Fit != byFit[j].Fit {
			return byFit[i].Fit > byFit[j].Fit
		}
		if byFit[i].Acc != byFit[j].Acc {
			return byFit[i].Acc > byFit[j].Acc
		}
		return byFit[i].Avail > byFit[j].Avail
	})
	out.TopFit = trimCells(byFit, 100)

	out.Scatter = downsampleScatter(byAcc, 4000)
	out.ByMode = aggregateAcc(cells, func(c cacheCell) string { return c.Mode })
	out.ByLayer = aggregateAcc(cells, func(c cacheCell) string { return c.Layer })
	out.ByDtype = aggregateAcc(cells, func(c cacheCell) string { return c.DType })
	out.ByFarm = aggregateAcc(cells, func(c cacheCell) string { return c.Farm })
	out.DtypeByLayer = nestedAccRank(cells,
		func(c cacheCell) string { return c.Layer },
		func(c cacheCell) string { return c.DType },
	)
	out.LayerByDtype = nestedAccRank(cells,
		func(c cacheCell) string { return c.DType },
		func(c cacheCell) string { return c.Layer },
	)
	out.FarmSeries = seriesBy(cells, func(c cacheCell) string { return c.Farm }, "farm", 24)
	out.ModeSeries = seriesBy(cells, func(c cacheCell) string { return c.Mode }, "mode", 16)
	out.LayerSeries = seriesBy(cells, func(c cacheCell) string { return c.Layer }, "layer", 16)
	return out
}

func trimCells(in []cacheCell, n int) []cacheCell {
	if len(in) <= n {
		return in
	}
	return in[:n]
}

func downsampleScatter(ranked []cacheCell, max int) []cacheCell {
	if len(ranked) <= max {
		return ranked
	}
	out := make([]cacheCell, 0, max)
	// keep top Acc + stride sample for density
	keepTop := max / 5
	if keepTop < 50 {
		keepTop = 50
	}
	out = append(out, ranked[:keepTop]...)
	step := len(ranked) / (max - keepTop)
	if step < 1 {
		step = 1
	}
	for i := keepTop; i < len(ranked) && len(out) < max; i += step {
		out = append(out, ranked[i])
	}
	return out
}

func aggregateAcc(cells []cacheCell, keyFn func(cacheCell) string) []accAgg {
	type bucket struct {
		n                            int
		sumAcc, sumFit, sumAvail     float64
		best                         cacheCell
	}
	m := map[string]*bucket{}
	for _, c := range cells {
		k := keyFn(c)
		if k == "" {
			k = "?"
		}
		b := m[k]
		if b == nil {
			b = &bucket{best: c}
			m[k] = b
		}
		b.n++
		b.sumAcc += c.Acc
		b.sumFit += c.Fit
		b.sumAvail += c.Avail
		if c.Acc > b.best.Acc || (c.Acc == b.best.Acc && c.Fit > b.best.Fit) {
			b.best = c
		}
	}
	out := make([]accAgg, 0, len(m))
	for k, b := range m {
		out = append(out, accAgg{
			Key: k, N: b.n,
			BestAcc: b.best.Acc, MeanAcc: b.sumAcc / float64(b.n),
			BestFit: b.best.Fit, MeanFit: b.sumFit / float64(b.n),
			MeanAvail: b.sumAvail / float64(b.n),
			BestID: b.best.ID, BestFarm: b.best.Farm,
			BestLayer: b.best.Layer, BestMode: b.best.Mode, BestDtype: b.best.DType,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].BestAcc != out[j].BestAcc {
			return out[i].BestAcc > out[j].BestAcc
		}
		return out[i].MeanAcc > out[j].MeanAcc
	})
	return out
}

// nestedAccRank groups by outer key, then ranks inner keys by Acc (e.g. layer → dtypes).
func nestedAccRank(cells []cacheCell, outerFn, innerFn func(cacheCell) string) []dtypeLayerBlock {
	groups := map[string][]cacheCell{}
	for _, c := range cells {
		o := outerFn(c)
		if o == "" {
			o = "?"
		}
		groups[o] = append(groups[o], c)
	}
	out := make([]dtypeLayerBlock, 0, len(groups))
	for o, rows := range groups {
		inner := aggregateAcc(rows, innerFn)
		best := 0.0
		if len(inner) > 0 {
			best = inner[0].BestAcc
		}
		out = append(out, dtypeLayerBlock{Key: o, N: len(rows), Best: best, Rows: inner})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Best != out[j].Best {
			return out[i].Best > out[j].Best
		}
		return out[i].Key < out[j].Key
	})
	return out
}

func seriesBy(cells []cacheCell, keyFn func(cacheCell) string, kind string, topK int) []accSeries {
	type lrBucket struct {
		n int
		sumAcc, sumAvail, bestAcc, bestFit float64
	}
	type seriesAcc struct {
		bestAcc float64
		byLR    map[float64]*lrBucket
		labels  map[float64]string
	}
	m := map[string]*seriesAcc{}
	for _, c := range cells {
		k := keyFn(c)
		if k == "" {
			continue
		}
		s := m[k]
		if s == nil {
			s = &seriesAcc{byLR: map[float64]*lrBucket{}, labels: map[float64]string{}}
			m[k] = s
		}
		if c.Acc > s.bestAcc {
			s.bestAcc = c.Acc
		}
		b := s.byLR[c.LR]
		if b == nil {
			b = &lrBucket{}
			s.byLR[c.LR] = b
			s.labels[c.LR] = c.LRLabel
		}
		b.n++
		b.sumAcc += c.Acc
		b.sumAvail += c.Avail
		if c.Acc > b.bestAcc {
			b.bestAcc = c.Acc
		}
		if c.Fit > b.bestFit {
			b.bestFit = c.Fit
		}
	}
	type ranked struct {
		name string
		s    *seriesAcc
	}
	list := make([]ranked, 0, len(m))
	for k, s := range m {
		list = append(list, ranked{k, s})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].s.bestAcc > list[j].s.bestAcc })
	if len(list) > topK {
		list = list[:topK]
	}
	out := make([]accSeries, 0, len(list))
	for _, it := range list {
		lrs := make([]float64, 0, len(it.s.byLR))
		for lr := range it.s.byLR {
			lrs = append(lrs, lr)
		}
		sort.Float64s(lrs)
		pts := make([]accSeriesPoint, 0, len(lrs))
		for _, lr := range lrs {
			b := it.s.byLR[lr]
			lbl := it.s.labels[lr]
			if lbl == "" {
				lbl = report.FormatLR(lr)
			}
			pts = append(pts, accSeriesPoint{
				LR: lr, LRLabel: lbl, N: b.n,
				BestAcc: b.bestAcc, MeanAcc: b.sumAcc / float64(b.n),
				BestFit: b.bestFit, MeanAvail: b.sumAvail / float64(b.n),
			})
		}
		out = append(out, accSeries{Name: it.name, Kind: kind, Points: pts})
	}
	return out
}

func (c *cacheIndex) Board() AccBoard {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.board
}

func (c *cacheIndex) Compare(title string) report.CompareReport {
	c.mu.RLock()
	tides := append([]report.NamedTideReport(nil), c.tides...)
	c.mu.RUnlock()
	if title == "" {
		title = "test53 cache Acc compare"
	}
	return report.BuildCompare(title, tides)
}

func (c *cacheIndex) Stats() (farms, cells int, root string, at time.Time) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.farms), len(c.cells), c.root, c.loadedAt
}
