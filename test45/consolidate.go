package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

const KindHier CellKind = "hier"

type trainedNet struct {
	job    job
	stack  *parallel.Stack
	result *ArchResult
}

type familySpec struct {
	name  string
	kinds []CellKind
}

func consolidationFamilies() []familySpec {
	return []familySpec{
		{name: "dense-like", kinds: []CellKind{KindDense, KindResidual, KindSequential, KindMetacognition, KindSwiGLU}},
		{name: "conv", kinds: []CellKind{KindCNN1, KindCNN2, KindCNN3, KindConvT1, KindConvT2, KindConvT3}},
		{name: "seq", kinds: []CellKind{KindMHA, KindLSTM, KindRNN, KindMamba, KindGDN}},
		{name: "norm", kinds: []CellKind{KindLayerNorm, KindRMSNorm, KindSoftmax, KindKMeans}},
	}
}

func hierName(nHemi int) string {
	return "hier/" + camName(nHemi)
}

// pickSurvivors keeps nets at nHemi whose TrainPix beat keepMin, capped at keepTop.
// Dead ~5% cluster is filtered. Always keep at least two (or one) so plumbing has a merge.
func pickSurvivors(nets []trainedNet, nHemi int, keepMin float64, keepTop int) []trainedNet {
	var at []trainedNet
	for _, n := range nets {
		if n.stack == nil || n.result == nil || n.result.Err != "" {
			continue
		}
		if n.job.kind == KindHier || n.job.nHemi != nHemi {
			continue
		}
		at = append(at, n)
	}
	sort.Slice(at, func(i, j int) bool {
		return at[i].result.Train.MeanPixel > at[j].result.Train.MeanPixel
	})
	if keepTop < 1 {
		keepTop = 8
	}
	var kept []trainedNet
	for _, n := range at {
		if len(kept) >= keepTop {
			break
		}
		if n.result.Train.MeanPixel+1e-9 >= keepMin {
			kept = append(kept, n)
		}
	}
	if len(kept) == 0 {
		lim := 2
		if len(at) < lim {
			lim = len(at)
		}
		kept = append(kept, at[:lim]...)
	}
	return kept
}

func cloneSandwich(kind CellKind, in, hidden, out, nHemi int, mode parallel.TrainMode, src *parallel.Stack) (*parallel.Stack, error) {
	dst, err := buildNativeCameral(kind, in, hidden, out, nHemi, mode)
	if err != nil {
		return nil, err
	}
	if err := copyOpWeights(dst, src); err != nil {
		return nil, fmt.Errorf("clone %s n=%d: %w", kind, nHemi, err)
	}
	return dst, nil
}

func stackStemMidHead(s *parallel.Stack) (stem, mid, head any, err error) {
	if s == nil || len(s.Children) < 3 {
		return nil, nil, nil, fmt.Errorf("sandwich needs stem/mid/head")
	}
	return s.Children[0], s.Children[1], s.Children[2], nil
}

// buildHierSandwich clones survivor mids, groups them by family (inner avg),
// then CombineFilter across families (outer plumbing gate). Stem/head come
// from the best TrainPix survivor so adapters already map 902-d.
func buildHierSandwich(survivors []trainedNet, in, hidden, out int, mode parallel.TrainMode) (stack *parallel.Stack, kept, note string, err error) {
	if len(survivors) == 0 {
		return nil, "", "", fmt.Errorf("no survivors")
	}
	best := survivors[0]
	for _, s := range survivors[1:] {
		if s.result.Train.MeanPixel > best.result.Train.MeanPixel {
			best = s
		}
	}
	bestClone, err := cloneSandwich(best.job.kind, in, hidden, out, best.job.nHemi, mode, best.stack)
	if err != nil {
		return nil, "", "", fmt.Errorf("clone best %s: %w", best.job.kind, err)
	}
	stem, _, head, err := stackStemMidHead(bestClone)
	if err != nil {
		return nil, "", "", err
	}

	type famPack struct {
		spec  familySpec
		nets  []trainedNet
		score float64
	}
	var packs []famPack
	used := map[CellKind]bool{}
	for _, fam := range consolidationFamilies() {
		var members []trainedNet
		sum := 0.0
		for _, s := range survivors {
			for _, k := range fam.kinds {
				if s.job.kind == k {
					members = append(members, s)
					sum += s.result.Train.MeanPixel
					used[s.job.kind] = true
					break
				}
			}
		}
		if len(members) == 0 {
			continue
		}
		packs = append(packs, famPack{spec: fam, nets: members, score: sum / float64(len(members))})
	}
	var leftover []trainedNet
	for _, s := range survivors {
		if !used[s.job.kind] {
			leftover = append(leftover, s)
		}
	}
	if len(leftover) > 0 {
		sum := 0.0
		for _, s := range leftover {
			sum += s.result.Train.MeanPixel
		}
		packs = append(packs, famPack{
			spec:  familySpec{name: "other", kinds: nil},
			nets:  leftover,
			score: sum / float64(len(leftover)),
		})
	}

	var familyOps []any
	var familyScores []float64
	var keptParts []string
	for _, p := range packs {
		op, names, err := mergeFamilyMids(p.nets, in, hidden, out, mode)
		if err != nil {
			return nil, "", "", fmt.Errorf("family %s: %w", p.spec.name, err)
		}
		familyOps = append(familyOps, op)
		familyScores = append(familyScores, p.score)
		keptParts = append(keptParts, fmt.Sprintf("%s[%s]", p.spec.name, names))
	}

	mid, err := filterMerge(hidden, familyOps, familyScores)
	if err != nil {
		return nil, "", "", err
	}
	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		return nil, "", "", err
	}
	s.Exec.Backend = core.BackendCPUTiled
	s.Exec.MultiCore = true
	s.Exec.TileSize = 32
	s.SyncChildExec()
	note = fmt.Sprintf("stem/head from %s (trainpix=%.1f); families=%d",
		jobName(best.job.kind, best.job.nHemi), best.result.Train.MeanPixel, len(familyOps))
	return s, joinComma(keptParts), note, nil
}

func mergeFamilyMids(nets []trainedNet, in, hidden, out int, mode parallel.TrainMode) (any, string, error) {
	var mids []any
	var names []string
	for _, n := range nets {
		cl, err := cloneSandwich(n.job.kind, in, hidden, out, n.job.nHemi, mode, n.stack)
		if err != nil {
			return nil, "", err
		}
		_, mid, _, err := stackStemMidHead(cl)
		if err != nil {
			return nil, "", err
		}
		mids = append(mids, mid)
		names = append(names, fmt.Sprintf("%s:%.1f", n.job.kind, n.result.Train.MeanPixel))
	}
	if len(mids) == 1 {
		return mids[0], names[0], nil
	}
	hemi, err := parallel.HemispheresFrom(parallel.Config{
		Dim: hidden, OutFeat: hidden, Branches: len(mids), Combine: parallel.CombineAvg,
	}, mids, nil)
	if err != nil {
		return nil, "", err
	}
	return hemi, joinComma(names), nil
}

func filterMerge(hidden int, ops []any, scores []float64) (any, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("filter merge empty")
	}
	if len(ops) == 1 {
		return ops[0], nil
	}
	gate, err := dense.NewConfigured[float32](hidden, len(ops), core.ActivationLinear,
		core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		return nil, fmt.Errorf("hier gate: %w", err)
	}
	seedGate(gate, scores)
	return parallel.HemispheresFrom(parallel.Config{
		Dim: hidden, OutFeat: hidden, Branches: len(ops), Combine: parallel.CombineFilter,
	}, ops, gate)
}

func seedGate(g *dense.Layer, scores []float64) {
	if g == nil || g.Weights == nil {
		return
	}
	n := len(scores)
	if n == 0 {
		return
	}
	max := scores[0]
	for _, s := range scores[1:] {
		if s > max {
			max = s
		}
	}
	if max <= 0 {
		max = 1
	}
	bias := make([]float64, n)
	for i, s := range scores {
		bias[i] = math.Log((s/max)*0.9 + 0.1)
	}
	g.Weights.Bias = bias
}

func joinComma(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += ", "
		}
		out += p
	}
	return out
}

func copyOpWeights(dst, src any) error {
	ss := dna.CollectStores(src)
	dd := dna.CollectStores(dst)
	if len(ss) != len(dd) {
		return fmt.Errorf("store count %d != %d", len(ss), len(dd))
	}
	for i := range ss {
		if ss[i] == nil || dd[i] == nil {
			continue
		}
		f, err := ss[i].FlattenF32()
		if err != nil {
			return fmt.Errorf("flatten store %d: %w", i, err)
		}
		if err := dd[i].SetFromF32(f); err != nil {
			return fmt.Errorf("set store %d: %w", i, err)
		}
		if len(ss[i].Bias) > 0 {
			dd[i].Bias = append([]float64(nil), ss[i].Bias...)
		}
	}
	return copyGDNTree(dst, src)
}

func copyGDNTree(dst, src any) error {
	switch s := src.(type) {
	case *gdn.Layer:
		d, ok := dst.(*gdn.Layer)
		if !ok || d == nil || s == nil {
			return fmt.Errorf("gdn type mismatch")
		}
		return copyGDN(d, s)
	case *parallel.Stack:
		d, ok := dst.(*parallel.Stack)
		if !ok || d == nil || s == nil {
			return fmt.Errorf("stack type mismatch")
		}
		if len(d.Children) != len(s.Children) {
			return fmt.Errorf("stack child count %d != %d", len(d.Children), len(s.Children))
		}
		for i := range s.Children {
			if err := copyGDNTree(d.Children[i], s.Children[i]); err != nil {
				return err
			}
		}
	case *parallel.Layer:
		d, ok := dst.(*parallel.Layer)
		if !ok || d == nil || s == nil {
			return fmt.Errorf("parallel type mismatch")
		}
		if len(d.Branches) != len(s.Branches) {
			return fmt.Errorf("branch count %d != %d", len(d.Branches), len(s.Branches))
		}
		for i := range s.Branches {
			if err := copyGDNTree(d.Branches[i], s.Branches[i]); err != nil {
				return err
			}
		}
	}
	return nil
}

func copyGDN(dst, src *gdn.Layer) error {
	copyBlob := func(dp **quant.Blob, sp *quant.Blob) error {
		if sp == nil {
			return nil
		}
		u, err := quant.Unpack(sp)
		if err != nil {
			return err
		}
		nb, err := quant.Pack(sp.Format, u, sp.Rows, sp.Cols)
		if err != nil {
			return err
		}
		*dp = nb
		return nil
	}
	if err := copyBlob(&dst.InQKV, src.InQKV); err != nil {
		return err
	}
	if err := copyBlob(&dst.InZ, src.InZ); err != nil {
		return err
	}
	if err := copyBlob(&dst.InB, src.InB); err != nil {
		return err
	}
	if err := copyBlob(&dst.InA, src.InA); err != nil {
		return err
	}
	if err := copyBlob(&dst.Out, src.Out); err != nil {
		return err
	}
	dst.ConvWeight = append([]float32(nil), src.ConvWeight...)
	dst.ALog = append([]float32(nil), src.ALog...)
	dst.DtBias = append([]float32(nil), src.DtBias...)
	dst.NormGamma = append([]float32(nil), src.NormGamma...)
	return nil
}
