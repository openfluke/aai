# Failed cheap-credit notes

Graveyard for “backprop but cheaper” recipes on the native-cam sine bench.
Code is **gone**. Do not revive. Keep TweenSplit.

Board: `apps/aai/test41_w_native_cam`, 10s, freq switch 2.5s (`1→2→3→4`),
Dense `10→32→32→1` + Bi/Tri/Quad `Dense → Hemispheres(n) → Dense`.
StepBP owns this race (~75–77% Acc). Split is the cheap ceiling (~30–34%).

Everyone starts from the same MSE gap:

\[
g_y=\frac{2}{d}(y-t)
\]

Backprop then walks \(g \leftarrow J^\top g\) down the sandwich. These did not.

13% flat + Cons ~75% + Stab ~98% = **collapsed job** (wrong and stable).
Not a new family. On Stack, Step\* and non-Step share one update — if they
diverge to 13%, the job died.

---

## Still shipping (for contrast)

### Tween — broadcast

\[
g_i = P(g_y),\quad \eta\leftarrow\eta/2
\]

Same output gap, resized onto every leaf. No \(1/N\). ~11% flat on sine.
Kept as the “this is not credit” baseline.

### TweenSplit — best cheap

\[
g_i = \frac{1}{N} P(g_y),\quad
\Delta W_i = -\eta\cdot\mathrm{localBackward}(g_i)
\]

\(N\) = trainable leaves. \(P\) = copy or block-average onto leaf post shape.
Local \(W^\top(g\odot\mathrm{act}')\) only — no chain into the next leaf.

Dense ~34%, Bi/Tri/Quad ~30–31%. Beats Tween. Does not beat StepBP.

`TrainStackMSE(..., ModeTweenSplit, lr)` · `layers/parallel/tween_split.go`

---

## Dead

### TweenHead

Head gets the full MSE gap. Hidden leaves get \(P(g_y)/\sqrt{H}\).

\[
g_{\mathrm{head}}=g_y,\qquad g_{\mathrm{hidden}}=\frac{P(g_y)}{\sqrt{H}}
\]

Idea: last leaf is “the output,” so give it real credit; shrink the rest.
Head here is the **output Dense**, not the stem. Still trained every leaf.

| Arch | Acc | vs Split | vs StepBP |
|------|----:|----------|-----------|
| Dense | 41% | Split 31% | StepBP ~77% |
| Bicameral | **13% flat** | dead | dead |
| Tri / Quad | ≈ Split | no lift | no lift |

Avail stayed ~8%. Only Dense moved; cameral died. Equation did not hold.

### TweenTrace v1 — EMA on the gap

\[
f \leftarrow 0.9\,f + 0.1\,\frac{P(g_y)}{N}
\]

Train from \(f\), not the live \(1/N\) gap. First step seeded \(f\) from live.

Idea: fragments that collect over time, like a cheap eligibility trace.

**13% flat, every arch.** \(f\) is an EMA of *other samples’* output errors
while the sine phase (and freq) keeps changing. 90% stale. Washed out Split.

### TweenTrace v2 — Polyak on Split’s dW

Live Split backward, then velocity in **weight** space:

\[
dW = \mathrm{localBackward}\!\left(\tfrac{1}{N}P(g_y)\right),\qquad
v \leftarrow 0.9\,v + 0.1\,dW,\qquad
W \leftarrow W - \eta v
\]

First step = Split. Idea: “just add traces on top of Split,” not replace \(g_y\).

| Label | Acc | vs Split |
|-------|----:|----------|
| Dense/TweenTrace | 25% | Split 34% |
| Dense/StepTweenTrace | **13% flat** | dead |
| Bicameral both | **13% flat** | dead |
| Tri / Quad Trace | ~23% | Split ~30–31% |

Adapt/Avail worse. μ=0.9 lags the 2.5s freq switches. Momentum on a
non-Jacobian dW is laggy Split, not missing \(J^\top\).

---

## What was actually missing

\[
\frac{\partial L}{\partial h_{\mathrm{stem}}}
= J_{\mathrm{stem}}^\top J_{\mathrm{hemi}}^\top J_{\mathrm{head}}^\top g_y
\]

That product is StepBP / `BackwardStack` / TweenChain. Fragments, \(1/\sqrt{H}\),
and dW momentum do not build it. TweenChain already walks it (~62% Acc, BP-priced).

Do not try another ρ.
