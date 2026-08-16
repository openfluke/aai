# Test 48 — test41 credit modes × every layer × xor / sine / copy × dtypes

Did FastProxy / Sparse beat StepBP on a 10s **sine** sandwich? This is the
same Lucy board on **every welvet layer kind**, on the toys those sandwiches
are supposed to fit, through **tricameral**, across **FormatNone weight
dtypes** (`core.AllDTypes`, 34). Activations stay **float32** (weight dtype ⊥
act `Tensor[T]`).

Not ARC. Not mesh.

| Task | What | Acc | SoftAcc |
|------|------|-----|---------|
| `xor` | 2-bit XOR in a 16-d vector | threshold 0.5 | `SoftAccProb` (scale 1) |
| `sine` | Lucy sine, freq `1→2→3→4` | \|err\| < 0.15 | `SoftAccOne` (scale 0.10) |
| `copy` | random 16-bit identity | bit match | `SoftAccProb` |

Arch: Dense stem → mid → Dense head. Mid is one Op (`-cam-min 1`) or
`Hemispheres(n)` for Bi / Tri. Spatial/seq kinds are View-wrapped like test47.

## Modes

`-modes all` (default) is every **Stack** mode from test41_w_native_cam
(16). Mesh* is skipped. On a Sandwich, Step\* and non-Step of the same family
share one update — they still run as separate jobs so you can see they match.

| Token | Mode | Credit |
|-------|------|--------|
| `stepbp` | StepBP | **Backprop** (`BackwardStack` + SGD) |
| `normalbp` | NormalBP | Same family as StepBP on Stack |
| `tween` / `steptween` | Tween | Broadcast \(P(g_y)\), half LR |
| `tweenchain` / `steptweenchain` | TweenChain | Chain rule (= BP on Stack) |
| `tweensplit` / `steptweensplit` | TweenSplit | \(1/N\,P(g_y)\) |
| `tweenalt` / `steptweenalt` | TweenAlt | Split then Tween |
| `headproxy` | HeadProxy | Head \(J^\top\); hidden `dW` only |
| `linear` | Linear | \(W^\top\) walk, skip act′ |
| `fastproxy` | FastProxy | \(W_{\mathrm{head}}^\top g_y\), all `dW` only |
| `linearcache` | LinearCache | **Dead on sine** (stale cache). Included for A/B. |
| `proxyasync` | HeadProxyAsync | Hidden use proxy from \(T-1\) |
| `sparse` | Sparse | Head + one rotating hidden |

Split-family jobs use `OpenSplitTape` (infer collect **is** the train tape).

Lucy (same as test41): **Acc**, SoftAcc, Avail = Infer/(Infer+Train), AdaptPct
(sine switches), Tput, Score = Tput × Avail × SoftAcc / 10_000.

Default `-duration 2s` is the short perm race. `-dtypes all` (default) is
**34×** the old float32 board (~98k jobs if layers/modes/tasks/cams are also
`all`). Pin storage for the comparable Lucy board:

```bash
go run . -dtypes float32 -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
```

`-workers 1` is Lucy-honest (Score/Avail). Default workers = NumCPU finishes
the full sweep faster but **shares CPU** so Avail/Score drop.

GDN mids have no Store dtype axis (`SetDType` is a no-op there); stem/head
Dense still convert. SIMD DotTile is **Dense + float32** only — other dtypes
use CPU tiled so backend is not a hidden axis.

## Run

```bash
cd apps/aai/test48

# dense only, headline modes, float32, 2s
go run . -dtypes float32 -layers dense -modes stepbp,tweensplit,headproxy,linear,fastproxy,proxyasync,sparse

# every layer, 1..3 cameral, all 16 modes, all 34 dtypes (huge)
go run . 

# xor + sine, CNN + Dense, Bi/Tri, f32+f16
go run . -dtypes float32,float16 -layers dense,cnn1,residual -tasks xor,sine -cam-min 1 -camerals 3 -duration 2s

# Lucy-honest 10s (slow) — pin float32
go run . -dtypes float32 -layers dense -modes stepbp,fastproxy,sparse -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-layers` | `all` | including Dense; every kind from test47 |
| `-camerals` | `3` | cap **tricameral** |
| `-cam-min` | `1` | 1 = single mid Op (“Dense” arch name) |
| `-only` | `0` | pin exact hemi count |
| `-duration` | `2s` | wall per job |
| `-switch` | duration/4 | sine frequency switch |
| `-adapt-windows` | `4` | AdaptPct pulse (10s race: 10) |
| `-modes` | `all` | 16 test41 stack modes |
| `-dtypes` | `all` | 34 `core.AllDTypes` (weight storage; act stays f32). Pin `float32` for the old board. |
| `-tasks` | `xor,sine,copy` | |
| `-hidden` | `32` | |
| `-lr` | `0.05` | test41 sine used 0.01; toys need 0.05 |
| `-workers` | NumCPU | `1` for comparable Score |
| `-alt-times` | `1` | TweenAlt pairs |

Writes `test48_results.json` (gitignored).

PDF report (tide / live_mnist axes — Acc, SoftAcc, Avail, AdaptPct, Score,
MobileScore, Pareto, vs StepBP):

```bash
.venv/bin/python report.py                          # uses test48_results.json
# first run streams the 1.5GB JSON (windows stripped → test48_flat.pkl)
```

Prints a Score-sorted table **per task × dtype**, then **vs StepBP** deltas for
FastProxy / Sparse / HeadProxy / Linear (matched inside that dtype).

Do not write “we beat backprop” from a 2s XOR lottery. Dense sine at
`-dtypes float32 -duration 10s -workers 1` is the comparable board.

The result write-up below is the **float32** 2s / 12-worker sweep (2880 jobs),
before the dtype axis. Re-run with `-dtypes float32` to reproduce it.



Everyone starts from the same **MSE gap** at the stack output \(y\), target \(t\), width \(d\):

\[
g_y=\frac{2}{d}(y-t)
\]

SGD on a leaf is always

\[
W\leftarrow W-\eta\,\mathrm{d}W
\]

The modes only change **how \(\mathrm{d}W\) is built**. \(P(\cdot)\) is `projectGap`: resize a vector onto that leaf’s post shape (block-average / nearest). **Not** a Jacobian.

A sandwich has \(N\) trainable leaves (stem, hemispheres, head, …). On Stack/Parallel, **Step\* and non-Step of the same family are the same update**. Mesh\* is the Grid/volumetric scheduler of that family (test41 Dense only).

---

### Backprop — `NormalBP` / `StepBP` / `MeshBP`

Real chain rule. One reverse pass:

\[
g_{\ell-1}=J_\ell^\top g_\ell
=W_\ell^\top\bigl(g_\ell\odot\mathrm{act}'(\mathrm{pre}_\ell)\bigr)
\]

\[
\mathrm{d}W_\ell=g_\ell^{\mathrm{pre}}\,x_\ell^\top
\]

Head gets \(g_y\). Stem gets \(J_{\mathrm{stem}}^\top J_{\mathrm{hemi}}^\top J_{\mathrm{head}}^\top g_y\). Each \(W^\top\) is the next layer’s \(g\). SIMD GEMV.

---

### TweenChain — `TweenChain` / `StepTweenChain` / `MeshTweenChain`

Same math as backprop **on a Sandwich** (`BackwardStack` + SGD). Different name / Grid scheduling on test41. Not a cheaper Jacobian.

---

### Tween — `Tween` / `StepTween` / `MeshTween`

No chain. Broadcast the **output** gap onto every leaf. Half LR.

\[
g_i=P(g_y),\qquad \eta\leftarrow\eta/2
\]

\[
\mathrm{d}W_i=\mathrm{localBackward}(g_i,x_i)
\]

Blind to \(W^\top\) sign. Sine Acc ~11%.

---

### TweenSplit — `TweenSplit` / `StepTweenSplit`

Same broadcast, split even:

\[
g_i=\frac{1}{N}P(g_y)
\]

Still not \(J^\top\). Dense ~33–35% on sine. Cheap ceiling.

---

### TweenAlt — `TweenAlt` / `StepTweenAlt`

Per sample, `AltTimes` times (default 1):

1. Split update from live \(g_y\)
2. Re-forward, new \(g_y'\)
3. Tween update from \(g_y'\) (half LR)

Split↔Tween ping-pong. Extra forwards → Avail dies.

---

### HeadProxy — `TweenSplitHeadProxy`

Head: **full** local backward (act′ and \(W^\top\)):

\[
g_{\mathrm{proxy}}=J_{\mathrm{head}}^\top g_y
=W_{\mathrm{head}}^\top\bigl(g_y\odot\mathrm{act}'(\mathrm{pre}_{\mathrm{head}})\bigr)
\]

Hidden \(i=1\ldots N-1\): **`dW` only** (no discarded \(W^\top\)):

\[
g_i=\frac{1}{N-1}P(g_{\mathrm{proxy}}),\qquad
\mathrm{d}W_i=g_i\,x_i^\top
\]

One real \(J_{\mathrm{head}}^\top\). Hemispheres do **not** get \(J_{\mathrm{hemi}}^\top\).

---

### FastProxy — `TweenSplitFastProxy`

Proxy **skips act′**. Pure SIMD \(W^\top\):

\[
g_{\mathrm{proxy}}=W_{\mathrm{head}}^\top g_y
\]

Head still uses act′ for **its own** `dW`. Hidden: same \(1/(N-1)\,P(g_{\mathrm{proxy}})\) `dW`-only as HeadProxy. No wait on head’s full backward. This is the full-net Score/Acc win on the sine/copy boards.

---

### HeadProxyAsync — `TweenSplitHeadProxyAsync`

Same FastProxy-style linearized proxy, **one sample late**:

\[
g_{\mathrm{hidden}}^{(T)}=\frac{1}{N-1}P\bigl(g_{\mathrm{proxy}}^{(T-1)}\bigr)
\]

Head computes \(g_{\mathrm{proxy}}^{(T)}=W_{\mathrm{head}}^\top g_y^{(T)}\) for next step. First sample seeds live. **Not EMA.**

---

### Linear — `TweenSplitLinear`

Affine chain, **skip \(\odot\mathrm{act}'\)**. Never form \(\tilde W=W_{\mathrm{head}}W_{\mathrm{hemi}}\) (that is \(O(N^3)\)).

Walk **matvecs** reverse through the tree. Hemispheres **share** the same down-going vector (siblings, not a product of each other):

\[
g_{\mathrm{head}}^{\downarrow}=g_y
\]
\[
g_{\mathrm{hemi}}^{\downarrow}=W_{\mathrm{head}}^\top g_y
\]
\[
g_{\mathrm{stem}}^{\downarrow}=\sum_{\mathrm{hemi}}W_{\mathrm{hemi}}^\top g_{\mathrm{hemi}}^{\downarrow}
\]

Then every leaf:

\[
g_i=\frac{1}{N}P(g_i^{\downarrow}),\qquad
\mathrm{d}W_i=g_i\,x_i^\top
\]

Same \(O(N^2)\) class as backprop, but you still pay every \(W^\top\) **and** every `dW`, so Score stays StepBP-class unless Acc is way up.

---

### LinearCache — `TweenSplitLinearCache` — **dead**

Every 20 steps: full Linear walk, cache per-leaf \(g_i^{\downarrow}\). In between:

\[
g_i\leftarrow g_i^{\mathrm{cache}}\cdot\frac{\|g_y\|_{\mathrm{live}}}{\|g_y\|_{\mathrm{cache}}}
\]

Norm scaling cannot recover sign flips after a frequency switch. Dense sine → 0%.

---

### Sparse — `TweenSplitSparse`

Head always. **One** rotating hidden leaf \(k=t \bmod (N-1)\):

\[
g_{\mathrm{proxy}}=W_{\mathrm{head}}^\top g_y
\]
\[
\mathrm{d}W_{\mathrm{head}}=g_y\,x_{\mathrm{head}}^\top,\qquad
\mathrm{d}W_k=P(g_{\mathrm{proxy}})\,x_k^\top
\]

Other leaves: \(\mathrm{d}W=0\) this sample. Real FLOP cut → Avail 25–40% → Lucy Score explodes. Acc held on sine/copy for some layers; not a smaller big-O than backprop.

---

### Local kernels (Dense)

\[
\begin{aligned}
\mathrm{LinearGradIn}:&\quad gx=W^\top g &&\text{(no act′)}\\
\mathrm{GradWOnly}:&\quad \mathrm{d}W=(g\odot\mathrm{act}'(\mathrm{pre}))\,x^\top &&\text{(no }W^\top\text{)}\\
\mathrm{Backward}:&\quad gx\text{ and }\mathrm{d}W &&\text{(both)}
\end{aligned}
\]

Non-Dense leaves (CNN/MHA/…) fall back to that Op’s `Backward`.

---

**Who is “backprop”:** StepBP / TweenChain.  
**Who injects one \(W^\top\):** HeadProxy (with act′), FastProxy (without).  
**Who walks every \(W^\top\):** Linear.  
**Who skips hidden GEMVs on purpose:** Sparse, and hidden HeadProxy/FastProxy `dW`-only.



The README is a **protocol**, not a result table. The numbers below are the sweep it produces: **2880 jobs**, 2 s, **12 workers**, every Stack mode × 20 layers × xor/sine/copy × Dense/Bi/Tri. Zero errors.

That is **not** the board the README says is comparable. Lucy-honest is `-duration 10s -workers 1` on Dense sine. Here Score/Avail are shared-CPU, Acc on Dense sine is already saturated, and XOR is a 2 s lottery. Read Acc and Score as two different questions.

---

### What Lucy is measuring

\[
\mathrm{Score}=\frac{\mathrm{Tput}\times\mathrm{Avail}\times\mathrm{SoftAcc}}{10\,000},\qquad
\mathrm{Avail}=\frac{\mathrm{Infer}}{\mathrm{Infer}+\mathrm{Train}}
\]

Hard **Acc** is a held-out eval after the clock. **SoftAcc** is live during the window. **AdaptPct** only fires on sine (freq `1→2→3→4`); xor/copy print 0. Score-sorted tables therefore crown whoever **skips GEMVs**, unless Acc actually collapsed.

---

### XOR — the lottery the README warns about

2-bit XOR stuffed in a 16-d vector, threshold 0.5. Acc clumps at **50 / 75 / 100**. Mean Acc: StepBP 74%, FastProxy 79%, Sparse **68%**, Tween 58%, LinearCache 42%, HeadProxyAsync **48%**.

FastProxy Acc-beats StepBP on only **13%** of cells (ties at 75 don’t count). Sparse Acc-beats on **5%**. Sparse still **Score-wins 100%** of cells because Avail is ~48% vs BP ~12%. Top Score is `layernorm/Bicameral/Sparse` **13782** at 100% Acc — a duty-clock spike, not a solver.

HeadProxyAsync is dead on XOR (mean Acc 48%): last sample’s proxy is the wrong sign for a discrete flip. Do not write “we beat backprop” from this task. The README is right.

---

### Sine — the only task with AdaptPct

Freq switches every 500 ms. Hard Acc can sit at 100% while SoftAcc is 30–50%, because Soft is **during** the jump.

| Mode | mean Acc | Soft | Avail | Adapt | Score |
|------|---------:|-----:|------:|------:|------:|
| StepBP / TweenChain | ~80% | ~34% | ~12% | ~33% | ~214 |
| Tween | 38% | 13% | 15% | 13% | 117 |
| TweenSplit | 55% | 23% | 28% | 21% | 426 |
| FastProxy | **87%** | 41% | 28% | 39% | 936 |
| Sparse | **88%** | 45% | **49%** | 43% | **2616** |
| HeadProxy | 84% | 31% | 28% | 29% | 837 |
| Linear | 80% | 36% | 20% | 36% | 555 |
| LinearCache | **32%** | 8% | 28% | **8%** | 194 |
| TweenAlt | 54% | 23% | **9%** | 21% | 48 |

**Dense sandwich (the README headline layer)** already hits **100% Acc** for StepBP, TweenChain, FastProxy, HeadProxy, Linear, Sparse. Acc cannot rank them. SoftAcc can: FastProxy Dense **67%** vs StepBP **53%** vs Sparse **49%**. Sparse still wins Score (**2359** vs StepBP **396**) because Avail is 20% vs 8% — it is updating one hidden leaf.

Tween is the blind broadcast: Dense Acc **41%**, Soft **10%**. TweenSplit’s cheap ceiling is real (~52% Acc on Dense). FastProxy is the thing that **clears** that ceiling with one \(W_{\mathrm{head}}^\top g_y\).

LinearCache is dead as advertised: sine mean Acc **31.5%**, Adapt **7.7%**. A few 100% Acc rows (`mamba`, `sequential`) have SoftAcc **2–5%** — hard-eval luck after a collapsed live fit, not a revival.

TweenAlt’s extra forward **kills Avail** (9%) so Score is last even when Acc is Split-like.

---

### Copy — Acc and Score finally disagree

Random 16-bit identity. AdaptPct = 0.

- **FastProxy Acc-beats StepBP on 72% of cells** (mean +6 Acc, +9 Soft). This is the cleanest “proxy > chain on this toy” in the file.
- **Sparse Acc-loses on average (−3)** while Score-winning **every** cell. Same story as the terminal paste: `rmsnorm/Dense/Sparse` Score **5737** at **72% Acc**; FastProxy on the same cell is **85% Acc / Score 3468**.
- HeadProxy Acc ~tied with BP; Linear Acc slightly up, Score still BP-class (still walks every \(W^\top\)).
- **cnn1 and convt1 are a chance floor for every mode including StepBP:** Acc **47.7%**, Soft **50%**, all 48 jobs. That is the View-wrapped sandwich, not credit. `softmax` is the same neighborhood (~50%). `kmeans` BP is **45%**. Those rows are not in the race.

rmsnorm / layernorm are the copy Score kings because they are cheap + Sparse skips most `dW`s. FastProxy is the Acc king on those same layers (rmsnorm 86% vs BP 73%; layernorm 87% vs 81%).

---

### vs StepBP (what the README prints)

Matched layer×arch, **Score Δ is almost always positive** for FastProxy / Sparse / HeadProxy / Linear because Avail rises. That is the duty clock.

Acc Δ is the honest column:

- **xor:** FastProxy +5 Acc, Sparse **−6**. Score win 100% / Acc win ~13% and 5%.
- **sine:** FastProxy +7, Sparse +7. Acc win only **20%** — the other 80% are ties at 100%. Score win 100%.
- **copy:** FastProxy **+6 Acc, 72% win rate**. Sparse **−3 Acc, 40% win rate**, Score still +2264.

TweenChain / NormalBP / StepTweenChain sit on top of StepBP (Acc Δ ~0, Score Δ ~0). Same family, same update. Residual |AccΔ| of 2–4 points is **12-way RNG + step count**, not a new Jacobian.

---

### Modes, mapped onto this file

- **StepBP / NormalBP / TweenChain** — real chain rule. Acc ceiling on toys. Avail ~11%. Score floor.
- **Tween** — \(P(g_y)\) broadcast. Sine Acc 38%. XOR lottery. Copy worse than Split.
- **TweenSplit** — \(1/N\,P(g_y)\). Cheap Acc ceiling ~55–70%. Score up from Avail, not from a better \(g\).
- **FastProxy** — \(W_{\mathrm{head}}^\top g_y\), all `dW`. Clears Split’s Acc ceiling on sine/copy. Score wins vs BP from Avail ~25–28%, not from skipping the whole net.
- **Sparse** — head + one rotating leaf. Lucy Score winner on every task. Acc held on sine (often 100% on Dense); **lost on XOR and often on copy**. Do not write “Sparse is better backprop.”
- **HeadProxy** — full \(J_{\mathrm{head}}^\top\). Acc near FastProxy on sine; SoftAcc worse (waits on act′). Score still Avail-driven.
- **HeadProxyAsync** — XOR disaster; sine/copy roughly FastProxy-lite. Stale proxy is not EMA.
- **Linear** — Acc ≈ BP, Score halfway (extra \(W^\top\) still paid).
- **LinearCache** — dead on sine; XOR Acc 42%; copy Acc 47%. Stale cache + \(\|g_y\|\) scale.
- **TweenAlt** — Split then Tween. Extra forward → Avail ~9% → Score last.

---

### What this file does **not** answer

The README’s first sentence: *Did FastProxy / Sparse beat StepBP on a 10 s sine sandwich?*

On **this** 2 s / 12-worker Dense sine board, Acc is 100% for all of them. SoftAcc says FastProxy > StepBP > Sparse. Score says Sparse >> FastProxy >> StepBP. That last ranking is Avail, and Avail is **not Lucy-honest** with 12 jobs on one CPU.

The comparable command is still:

```bash
go run . -layers dense -modes stepbp,fastproxy,sparse -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
```

Until that board, the true statements from test48 are: FastProxy can **match or beat StepBP Acc** on copy and on unsaturated sine layers; Sparse **wins the Lucy clock** and **does not** consistently win Acc; LinearCache and Tween stay dead; cnn1/convt1 copy is a floor, not a mode result.