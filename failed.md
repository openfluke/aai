# Failed cheap-credit notes

Graveyard for “backprop but cheaper” recipes on the native-cam sine bench.
Do not revive the **Dead** list. HeadProxy / Linear / FastProxy / Sparse
stayed — they actually inject \(W^\top\) (or skip hidden GEMVs on purpose).

Board: `apps/aai/test41_w_native_cam`, 10s, freq switch 2.5s (`1→2→3→4`),
Dense `10→32→32→1` + Bi/Tri/Quad `Dense → Hemispheres(n) → Dense`.
Latest opt-in race (one tape + Dense `dW`-only + `MatVecT`):

| Recipe | Dense Acc | Dense Score | vs StepBP Score 3914 |
|--------|----------:|------------:|----------------------|
| **Sparse** | 75% | **11215** | Avail 26% — fewer leaves |
| **FastProxy** | **83%** | **5222** | Acc + Score |
| HeadProxy | 72% | 4596 | Acc matched, Avail slightly up |
| HeadProxyAsync | 75% | 4387 | 1-sample delay held |
| Linear | **82%** | 4122 | Acc wins, walk still costs |
| StepBP | 73% | 3914 | — |
| Split | 36% | 2115 | ceiling |
| **LinearCache** | **14% → 0** | 675 | **dead** |

Cameral Score winner this run: **Tricameral/Sparse 15146** (Acc 84%, Avail 37%).
Do **not** write “Sparse is better backprop.” It updates head + one hidden.
Acc held on this sine. Score is Tput × Avail. FastProxy is the full-net win.

Everyone starts from the same MSE gap:

\[
g_y=\frac{2}{d}(y-t)
\]

13% flat + Cons ~75% + Stab ~98% = **collapsed job** (wrong and stable).
Not a new family. On Stack, Step\* and non-Step share one update — if they
diverge to 13%, the job died. (This run: Bicameral/Tricameral TweenSplit went
13% while StepTweenSplit stayed ~30%. Don’t scrap Split for that.)

---

## Still shipping

### Tween — broadcast (baseline, not credit)

\[
g_i = P(g_y),\quad \eta\leftarrow\eta/2
\]

~11% flat on sine.

### TweenSplit — old cheap ceiling

\[
g_i = \frac{1}{N} P(g_y)
\]

Dense ~33–35%. Cameral ~25–32% when it doesn’t collapse. Blind to \(W^\top\)
sign. Beats Tween. Beaten by HeadProxy / Linear.

### TweenSplitHeadProxy — keep

Head: full \(g_y\) + real local backward (\(J_{\mathrm{head}}^\top\)).
Hidden: \(\frac{1}{N-1}P(g_{\mathrm{proxy}})\) **`dW` only** (no discarded \(W^\top\)).

One-tape 10s vs StepBP:

| Arch | HeadProxy Acc | Score | StepBP Acc | Score |
|------|--------------:|------:|-----------:|------:|
| Dense | 72% | **4596** | 73% | 3914 |
| Bicameral | 68% | **2692** | 76% | 1778 |
| Tricameral | 62% | **1547** | 75% | 1162 |
| Quadcameral | 63% | **1149** | 76% | 721 |

Acc still trails StepBP on cameral. Score wins because Avail rose (~13–16%)
once the extra forwards died. Tiny lock on the head only.

### TweenSplitLinear — keep (Acc), Score still not the story

\[
g_i=\frac{1}{N}P(\tilde W_i^\top g_y)
\]

Dense \(W^\top\) chain, **skip** \(\odot\mathrm{act}'\). Hemispheres share
\(W_{\mathrm{head}}^\top g_y\) (siblings, not a product of each other).
One SIMD `MatVecT` walk — never \(W\times W\).

| Arch | Linear Acc | Score | StepBP Acc | Score |
|------|-----------:|------:|-----------:|------:|
| Dense | **82%** | 4122 | 73% | 3914 |
| Bicameral | 76% | 1961 | 76% | 1778 |
| Tricameral | 72% | 1098 | 75% | 1162 |
| Quadcameral | 72% | 655 | 76% | 721 |

Acc can match or beat StepBP on this leaky-ReLU sine. Linear still walks
**every** \(W^\top\) and still forms every `dW`, so Avail stays ≤ StepBP
(Dense 13.9% vs 15.7%). One tape pulled Score up to StepBP-class; FastProxy
is cheaper for the same Acc.

### TweenSplitFastProxy — keep (full-net win)

\[
g_{\mathrm{proxy}} = W_{\mathrm{head}}^\top g_y
\quad\text{(no act′)},\qquad
g_{\mathrm{hidden}} = \tfrac{1}{N-1}P(g_{\mathrm{proxy}})
\]

Head `dW` still uses act′. Hidden `dW` only. No head-backward lock.

| Arch | FastProxy Acc | Score | StepBP Acc | Score |
|------|--------------:|------:|-----------:|------:|
| Dense | **83%** | **5222** | 73% | 3914 |
| Bicameral | **81%** | **3008** | 76% | 1778 |
| Tricameral | **79%** | **2016** | 75% | 1162 |
| Quadcameral | 73% | **1210** | 76% | 721 |

This is the recipe that actually **beats StepBP Score with the whole net
updating**. Acc held (Dense/Bi/Tri beat StepBP Acc; Quad matched). Avail
~13–17%, same class as StepBP, not Sparse’s 26–39%.

### TweenSplitHeadProxyAsync — keep

Hidden use linearized proxy from sample \(T-1\); head computes \(T\).
**Not EMA.** Acc 73–75% all arches. Dense Score 4387 (beats StepBP, trails
FastProxy). 1-sample delay did not wash out.

### TweenSplitSparse — keep (FLOP lever, not a new Jacobian)

Head + **one** rotating hidden leaf per sample. That is why Avail hits
26–39% and Tput ~2–4× StepBP.

| Arch | Sparse Acc | Avail | Score | StepBP Score |
|------|-----------:|------:|------:|-------------:|
| Dense | 75% | 26% | **11215** | 3914 |
| Bicameral | 78% | 32% | **14033** | 1778 |
| Tricameral | **84%** | **37%** | **15146** | 1162 |
| Quadcameral | 80% | 39% | **14132** | 721 |

Acc held on this sine (the surprise). Score is the duty clock. Do not claim
a smaller big-O than backprop — you just skipped most `dW`s. Re-check on
ARC / harder tasks before making Sparse the default.

`ModeTweenSplitHeadProxy` / `Linear` / `FastProxy` / `HeadProxyAsync` /
`Sparse` · `layers/parallel/tween_split_w.go`

---

## Dead

### TweenSplitLinearCache — scrap (stale \(\tilde W\) direction)

Refresh the Linear walk every 20 steps; in between, scale the cached per-leaf
vectors by \(\|g_y\|_{\mathrm{live}}/\|g_y\|_{\mathrm{cache}}\).

**Why it failed:** same class as TweenTrace. Weights move; frequency
switches at 2.5s. A cached direction from sample \(T-20\) is not
\(W^\top g_y\) at \(T\). Scaling by gap *norm* cannot recover sign changes.

| Arch | Acc | vs Linear | vs StepBP |
|------|----:|-----------|-----------|
| Dense | **14% → 0% after 6s** | Linear 82% | dead |
| Bicameral | 24% | 76% | dead-ish |
| Tricameral | 25% | 72% | dead-ish |
| Quadcameral | 25% | 72% | dead-ish |

Dense Cons 55% + late zeros = collapsed job. Do not revive. Leave the
mode in the binary for A/B; do not put it on the default board.

### TweenSplitWScale — scrap (code gone)

Tried to replace the even \(1/N\) split with weight-size credit. Same \(P(g_y)\)
broadcast, different scalar:

\[
S = \sum_k \|W_k\|_F
\qquad
g_i = \left(\frac{\|W_i\|_F}{S}\right) P(g_y)
\qquad
\Delta W_i = -\eta\cdot\mathrm{localBackward}(g_i)
\]

\(\|W\|_F=\sqrt{\sum_{uv}W_{uv}^2}\). Idea: a leaf with bigger weights “did more,”
so it should eat more of the output gap.

**Why it failed:** Frobenius is a **positive scalar**. It cannot flip a sign.
If Head has negative weights, \(g_y\) still says “go up” while Hemi needs “go
down” — WScale still broadcasts that same \(g_y\), just louder or quieter.
Magnitude is not \(W^\top\). It can also starve a small-norm leaf (tiny
\(\|W\|_F/S\)) so that layer barely moves.

10s sine vs Split:

| Arch | WScale | vs Split |
|------|-------:|----------|
| Dense | **13% flat** | 33% |
| Bicameral | 30% | ≈ 32% |
| Tricameral | 29% | ≈ 30% |
| Quadcameral | **13% flat** | 29% |

Dead (13% floor) or Split-with-extra-math. Code removed from
`layers/parallel/tween_split_w.go` / `train_mode.go` / test41. Do not revive.

### TweenHead

Head gets the full MSE gap. Hidden leaves get \(P(g_y)/\sqrt{H}\).

\[
g_{\mathrm{head}}=g_y,\qquad g_{\mathrm{hidden}}=\frac{P(g_y)}{\sqrt{H}}
\]

Idea: last leaf is “the output,” so give it real credit; shrink the rest.
Head here is the **output Dense**, not the stem. Still trained every leaf.
**Not HeadProxy** — hidden still got a resized \(g_y\), not \(J_{\mathrm{head}}^\top g_y\).

| Arch | Acc | vs Split | vs StepBP |
|------|----:|----------|-----------|
| Dense | 41% | Split 31% | StepBP ~77% |
| Bicameral | **13% flat** | dead | dead |
| Tri / Quad | ≈ Split | no lift | no lift |

Avail stayed ~8%. Only Dense moved; cameral died.

### TweenTrace v1 — EMA on the gap

\[
f \leftarrow 0.9\,f + 0.1\,\frac{P(g_y)}{N}
\]

Train from \(f\), not the live \(1/N\) gap. First step seeded \(f\) from live.

**13% flat, every arch.** EMA of other samples’ output errors.

### TweenTrace v2 — Polyak on Split’s dW

\[
dW = \mathrm{localBackward}\!\left(\tfrac{1}{N}P(g_y)\right),\qquad
v \leftarrow 0.9\,v + 0.1\,dW
\]

| Label | Acc | vs Split |
|-------|----:|----------|
| Dense/TweenTrace | 25% | Split 34% |
| Dense/StepTweenTrace | **13% flat** | dead |
| Bicameral both | **13% flat** | dead |
| Tri / Quad Trace | ~23% | Split ~30–31% |

Momentum on a non-Jacobian dW is laggy Split.

---

## What was actually missing

\[
\frac{\partial L}{\partial h_{\mathrm{stem}}}
= J_{\mathrm{stem}}^\top J_{\mathrm{hemi}}^\top J_{\mathrm{head}}^\top g_y
\]

On this sandwich the expensive bit was **\(W^\top\)**, not a ρ and not
\(\|W\|_F\). HeadProxy injects one \(J_{\mathrm{head}}^\top\). FastProxy injects
\(W_{\mathrm{head}}^\top g_y\) without act′ and skips hidden \(W^\top\).
Linear injects the affine chain and skips act′. Sparse skips most leaves.
Fragments / \(1/\sqrt{H}\) / Frobenius / cached \(\tilde W\) do not.

Do not try another ρ. Do not revive WScale. Do not revive LinearCache.
