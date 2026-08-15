# Failed cheap-credit notes

Graveyard for “backprop but cheaper” recipes on the native-cam sine bench.
Do not revive the **Dead** list. HeadProxy / Linear stayed — they actually
inject \(W^\top\).

Board: `apps/aai/test41_w_native_cam`, 10s, freq switch 2.5s (`1→2→3→4`),
Dense `10→32→32→1` + Bi/Tri/Quad `Dense → Hemispheres(n) → Dense`.

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
Hidden: \(\frac{1}{N-1}P(g_{\mathrm{proxy}})\) with \(g_{\mathrm{proxy}}=J_{\mathrm{head}}^\top g_y\).

10s sine vs Split / StepBP:

| Arch | HeadProxy | Split | StepBP |
|------|----------:|------:|-------:|
| Dense | **67%** | 33–35% | 72% |
| Bicameral | **59%** | 32% | 77% |
| Tricameral | **48%** | 30% | 78% |
| Quadcameral | **57%** | 25–29% | 75% |

Avail ~7–8% (Split-like). Best **Score** of the Split family (Dense 1482 vs
Split 704 vs StepBP 4158). Tiny lock on the head only.

### TweenSplitLinear — keep (Acc), not Score

\[
g_i=\frac{1}{N}P(\tilde W_i^\top g_y)
\]

Dense \(W^\top\) chain, **skip** \(\odot\mathrm{act}'\). Hemispheres share
\(W_{\mathrm{head}}^\top g_y\) (siblings, not a product of each other).

| Arch | Linear | StepBP | Split |
|------|-------:|-------:|------:|
| Dense | **77%** | 72% | 33% |
| Bicameral | **75%** | 77% | 32% |
| Tricameral | **70%** | 78% | 30% |
| Quadcameral | **67%** | 75% | 29% |

Acc is StepBP-class on this leaky-ReLU sine (act′ is 0.01 or 1, so skipping it
is a mild lie). **Score still loses**: Avail 5–6% vs StepBP 10–16% (extra \(W^\top\)
walk). Dense Linear Score 959 vs StepBP 4158. Do not write “Linear beat
backprop” — it beat Split Acc, and matched StepBP Acc on Dense this run.

`ModeTweenSplitHeadProxy` / `ModeTweenSplitLinear` · `layers/parallel/tween_split_w.go`

---

## Dead

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
\(\|W\|_F\). HeadProxy injects one \(J_{\mathrm{head}}^\top\). Linear injects the
affine chain and skips act′. Fragments / \(1/\sqrt{H}\) / Frobenius do not.

Do not try another ρ. Do not revive WScale.
