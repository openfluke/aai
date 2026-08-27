package main

import (
	"math/rand"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/runtime/forward"
)

// dayroute — synthetic human day on an XY grid.
//
// Wake → bath → breakfast → work → lunch → gym → couch → sleep.
// Repeats for 5 days. Places drift a little each morning so the route
// "moves". Agent picks 1 of 6 actions: N/S/E/W / ACT / WAIT.

const (
	gridSize = 8
	nDays    = 5
	nActs    = 6 // N S E W ACT WAIT
	nSlots   = 8 // schedule length
	obsDim   = 24
	actN     = 0
	actS     = 1
	actE     = 2
	actW     = 3
	actDo    = 4
	actWait  = 5
)

type place struct{ x, y int }

type sample struct {
	x, y *core.Tensor[float32]
}

type dayWorld struct {
	rng     *rand.Rand
	day     int
	slot    int
	ax, ay  int
	places  []place
	names   []string
	steps   int
	doneDay int
}

func newDayWorld(seed int64) *dayWorld {
	w := &dayWorld{
		rng: rand.New(rand.NewSource(seed)),
		names: []string{
			"wake", "bath", "breakfast", "work",
			"lunch", "gym", "couch", "sleep",
		},
	}
	w.reshuffleDay()
	return w
}

func (w *dayWorld) basePlaces() []place {
	return []place{
		{1, 1}, // bed / wake
		{1, 3}, // bath
		{3, 1}, // kitchen breakfast
		{6, 6}, // desk
		{3, 2}, // kitchen lunch
		{6, 1}, // gym
		{3, 6}, // couch
		{1, 1}, // bed / sleep
	}
}

func (w *dayWorld) reshuffleDay() {
	base := w.basePlaces()
	w.places = make([]place, nSlots)
	for i, p := range base {
		dx := w.rng.Intn(3) - 1
		dy := w.rng.Intn(3) - 1
		w.places[i] = place{
			x: clamp(p.x+dx, 0, gridSize-1),
			y: clamp(p.y+dy, 0, gridSize-1),
		}
	}
	w.ax, w.ay = w.places[0].x, w.places[0].y
	w.slot = 0
}

func (w *dayWorld) goal() place {
	if w.slot < 0 || w.slot >= len(w.places) {
		return w.places[len(w.places)-1]
	}
	return w.places[w.slot]
}

func (w *dayWorld) obs() *core.Tensor[float32] {
	t := core.NewTensor[float32](1, obsDim)
	g := w.goal()
	t.Data[0] = float32(w.ax) / float32(gridSize-1)
	t.Data[1] = float32(w.ay) / float32(gridSize-1)
	t.Data[2] = float32(w.day) / float32(nDays-1)
	t.Data[3] = float32(w.slot) / float32(nSlots-1)
	t.Data[4] = float32(g.x) / float32(gridSize-1)
	t.Data[5] = float32(g.y) / float32(gridSize-1)
	t.Data[6] = float32(g.x-w.ax) / float32(gridSize-1)
	t.Data[7] = float32(g.y-w.ay) / float32(gridSize-1)
	if w.ax == g.x && w.ay == g.y {
		t.Data[8] = 1
	}
	if w.slot >= 0 && w.slot < nSlots {
		t.Data[9+w.slot] = 1
	}
	t.Data[17] = float32(w.doneDay) / float32(nDays)
	t.Data[18] = float32(w.steps%64) / 63
	return t
}

func (w *dayWorld) oracle() int {
	g := w.goal()
	dx, dy := g.x-w.ax, g.y-w.ay
	if dx == 0 && dy == 0 {
		return actDo
	}
	if abs(dx) >= abs(dy) {
		if dx > 0 {
			return actE
		}
		return actW
	}
	if dy > 0 {
		return actS
	}
	return actN
}

func (w *dayWorld) target(action int) *core.Tensor[float32] {
	y := core.NewTensor[float32](1, nActs)
	if action < 0 || action >= nActs {
		action = actWait
	}
	y.Data[action] = 1
	return y
}

func (w *dayWorld) step(a int) {
	w.steps++
	g := w.goal()
	switch a {
	case actN:
		w.ay = clamp(w.ay-1, 0, gridSize-1)
	case actS:
		w.ay = clamp(w.ay+1, 0, gridSize-1)
	case actW:
		w.ax = clamp(w.ax-1, 0, gridSize-1)
	case actE:
		w.ax = clamp(w.ax+1, 0, gridSize-1)
	case actDo:
		if w.ax == g.x && w.ay == g.y {
			w.slot++
			if w.slot >= nSlots {
				w.doneDay++
				w.day = (w.day + 1) % nDays
				w.reshuffleDay()
			}
		}
	}
}

func (w *dayWorld) sample() sample {
	return sample{x: w.obs(), y: w.target(w.oracle())}
}

func (w *dayWorld) advanceTeacher() sample {
	s := w.sample()
	w.step(w.oracle())
	return s
}

func scoreAct(post, target *core.Tensor[float32]) (hard, soft float64) {
	if post == nil || target == nil || post.Len() == 0 || target.Len() == 0 {
		return 0, 0
	}
	n := post.Len()
	if target.Len() < n {
		n = target.Len()
	}
	pred, gold := 0, 0
	for i := 1; i < n; i++ {
		if post.Data[i] > post.Data[pred] {
			pred = i
		}
		if target.Data[i] > target.Data[gold] {
			gold = i
		}
	}
	if pred == gold {
		hard = 100
	}
	soft = lucy.SoftAccProb(post.Data[gold], 1)
	return hard, soft
}

func evalRoute(st *parallel.Stack, grid *architecture.Grid, mode parallel.TrainMode, seed int64, steps int) (acc, soft, days float64) {
	w := newDayWorld(seed + 99)
	var hs, ss float64
	n := 0
	for i := 0; i < steps; i++ {
		s := w.sample()
		post := forwardOnce(st, grid, mode, s.x)
		if post != nil {
			h, so := scoreAct(post, s.y)
			hs += h
			ss += so
			n++
		}
		a := actWait
		if post != nil {
			a = argmax(post.Data)
		}
		w.step(a)
	}
	if n == 0 {
		return 0, 0, 0
	}
	return hs / float64(n), ss / float64(n), float64(w.doneDay)
}

func forwardOnce(st *parallel.Stack, grid *architecture.Grid, mode parallel.TrainMode, x *core.Tensor[float32]) *core.Tensor[float32] {
	if mode == parallel.ModeMeshBP || mode == parallel.ModeMeshTween || mode == parallel.ModeMeshTweenChain {
		if grid == nil {
			return nil
		}
		fwd, err := forward.Forward(grid, x)
		if err != nil || fwd == nil {
			return nil
		}
		return fwd.Output
	}
	_, p, err := parallel.ForwardStack(st, x)
	if err != nil {
		return nil
	}
	return p
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
