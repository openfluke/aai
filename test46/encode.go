package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ARC-AGI grids are 1×1 … 30×30, colors 0–9.
const (
	MaxGrid  = 30
	ColorMax = 9
	SizeFeat = 2 // [H/Max, W/Max] prefix on every vector
)

func vecDim(max int) int { return SizeFeat + max*max }

type Pair struct {
	Input  [][]int `json:"input"`
	Output [][]int `json:"output"`
}

type Task struct {
	ID    string
	Path  string
	Train []Pair `json:"train"`
	Test  []Pair `json:"test"`
}

func loadTasks(dir string, offset, limit int) ([]Task, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	if offset < 0 {
		offset = 0
	}
	if offset > len(matches) {
		offset = len(matches)
	}
	matches = matches[offset:]
	if limit > 0 && limit < len(matches) {
		matches = matches[:limit]
	}
	out := make([]Task, 0, len(matches))
	for _, p := range matches {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		var t Task
		if err := json.Unmarshal(raw, &t); err != nil {
			return nil, fmt.Errorf("parse %s: %w", p, err)
		}
		t.Path = p
		t.ID = strings.TrimSuffix(filepath.Base(p), ".json")
		if len(t.Train) == 0 || len(t.Test) == 0 {
			continue
		}
		out = append(out, t)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no ARC tasks in %s", dir)
	}
	return out, nil
}

func gridSize(g [][]int) (h, w int) {
	h = len(g)
	if h > 0 {
		w = len(g[0])
	}
	return h, w
}

func encodeGrid(g [][]int, max int) []float32 {
	h, w := gridSize(g)
	vec := make([]float32, vecDim(max))
	if max <= 0 {
		return vec
	}
	vec[0] = float32(h) / float32(max)
	vec[1] = float32(w) / float32(max)
	for r := 0; r < h && r < max; r++ {
		row := g[r]
		for c := 0; c < w && c < max && c < len(row); c++ {
			col := row[c]
			if col < 0 {
				col = 0
			}
			if col > ColorMax {
				col = ColorMax
			}
			vec[SizeFeat+r*max+c] = float32(col) / float32(ColorMax)
		}
	}
	return vec
}

func decodeSize(vec []float32, max int) (h, w int) {
	if max < 1 {
		max = MaxGrid
	}
	h = clampRound(vecAt(vec, 0)*float32(max), 1, max)
	w = clampRound(vecAt(vec, 1)*float32(max), 1, max)
	return h, w
}

func decodeGrid(vec []float32, h, w, max int) [][]int {
	if max < 1 {
		max = MaxGrid
	}
	if h < 1 {
		h = 1
	}
	if w < 1 {
		w = 1
	}
	if h > max {
		h = max
	}
	if w > max {
		w = max
	}
	g := make([][]int, h)
	for r := 0; r < h; r++ {
		row := make([]int, w)
		for c := 0; c < w; c++ {
			v := vecAt(vec, SizeFeat+r*max+c)
			row[c] = clampRound(v*float32(ColorMax), 0, ColorMax)
		}
		g[r] = row
	}
	return g
}

func vecAt(vec []float32, i int) float32 {
	if i < 0 || i >= len(vec) {
		return 0
	}
	return vec[i]
}

func clampRound(v float32, lo, hi int) int {
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return lo
	}
	n := int(math.Round(float64(v)))
	if n < lo {
		return lo
	}
	if n > hi {
		return hi
	}
	return n
}

func gridsEqual(a, b [][]int) bool {
	ah, aw := gridSize(a)
	bh, bw := gridSize(b)
	if ah != bh || aw != bw {
		return false
	}
	for r := 0; r < ah; r++ {
		for c := 0; c < aw; c++ {
			if a[r][c] != b[r][c] {
				return false
			}
		}
	}
	return true
}

func pixelAcc(pred, gold [][]int) float64 {
	gh, gw := gridSize(gold)
	if gh == 0 || gw == 0 {
		return 0
	}
	ph, pw := gridSize(pred)
	match := 0
	total := gh * gw
	for r := 0; r < gh; r++ {
		for c := 0; c < gw; c++ {
			if r < ph && c < pw && pred[r][c] == gold[r][c] {
				match++
			}
		}
	}
	return 100 * float64(match) / float64(total)
}

func colorSoftAcc(pred, gold []float32, max int) float64 {
	n := vecDim(max)
	if len(pred) < n || len(gold) < n {
		n = len(pred)
		if len(gold) < n {
			n = len(gold)
		}
	}
	if n <= SizeFeat {
		return 0
	}
	sum := 0.0
	cells := n - SizeFeat
	for i := SizeFeat; i < n; i++ {
		// scale=1.0 so off-by-one color (~1/9) still scores ~89 SoftAcc
		d := math.Abs(float64(pred[i] - gold[i]))
		a := 100 * (1 - d)
		if a < 0 {
			a = 0
		}
		if a > 100 {
			a = 100
		}
		sum += a
	}
	return sum / float64(cells)
}
