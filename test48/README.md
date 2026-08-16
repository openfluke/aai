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

Lucy measuring (same as test41; equations under **What Lucy is measuring**):
hard Acc, SoftAcc, Avail, AdaptPct, Tput, Score.

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

PDFs (matplotlib, same `.venv`):

```bash
.venv/bin/python report.py       # Lucy tables / heatmaps → test48_report.pdf
.venv/bin/python code_pdf.py     # every test48 *.go     → test48_code.pdf
```

`report.py` streams the JSON once (windows stripped → `test48_flat.pkl`), then
the honesty grids: every train mode × every dtype × every layer, Acc and AccΔ
vs StepBP.

Do not write “we beat backprop” from a 2s XOR lottery. Dense sine at
`-dtypes float32 -duration 10s -workers 1` is the comparable board.

---

## Shared math (every mode)

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

Never form \(\tilde W=W_{\mathrm{head}}W_{\mathrm{hemi}}\). That is \(O(N^3)\). Credit walks **matvecs**.

---

### Backprop — `NormalBP` / `StepBP`

Real chain rule. One reverse pass:

\[
g_{\ell-1}=J_\ell^\top g_\ell
=W_\ell^\top\bigl(g_\ell\odot\mathrm{act}'(\mathrm{pre}_\ell)\bigr)
\]

\[
\mathrm{d}W_\ell=g_\ell^{\mathrm{pre}}\,x_\ell^\top
\]

Head gets \(g_y\). Stem gets \(J_{\mathrm{stem}}^\top J_{\mathrm{hemi}}^\top J_{\mathrm{head}}^\top g_y\). Each \(W^\top\) is the next layer’s \(g\). SIMD GEMV. **This is the rival**, not Lucy Score.

---

### TweenChain — `TweenChain` / `StepTweenChain`

Same math as backprop **on a Sandwich** (`BackwardStack` + SGD). Different name. Not a cheaper Jacobian.

---

### Tween — `Tween` / `StepTween`

No chain. Broadcast the **output** gap onto every leaf. Half LR.

\[
g_i=P(g_y),\qquad \eta\leftarrow\eta/2
\]

\[
\mathrm{d}W_i=\mathrm{localBackward}(g_i,x_i)
\]

Blind to \(W^\top\) sign. Sine Acc collapses (~11–27% depending on dtype mix).

---

### TweenSplit — `TweenSplit` / `StepTweenSplit`

Same broadcast, split even:

\[
g_i=\frac{1}{N}P(g_y)
\]

Still not \(J^\top\). Cheap Acc ceiling. Score rises because the update is cheap (Avail), not because \(g\) is better.

---

### TweenAlt — `TweenAlt` / `StepTweenAlt`

Per sample, `AltTimes` times (default 1):

1. Split update from live \(g_y\)
2. Re-forward, new \(g_y'\)
3. Tween update from \(g_y'\) (half LR)

Split↔Tween ping-pong. Extra forwards → Avail dies → Score last.

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

Head still uses act′ for **its own** `dW`. Hidden: same \(1/(N-1)\,P(g_{\mathrm{proxy}})\) `dW`-only as HeadProxy. No wait on head’s full backward. This is DFA with \(B:=W_{\mathrm{head}}^\top\), not a learned random \(B\).

---

### HeadProxyAsync — `TweenSplitHeadProxyAsync`

Same FastProxy-style linearized proxy, **one sample late**:

\[
g_{\mathrm{hidden}}^{(T)}=\frac{1}{N-1}P\bigl(g_{\mathrm{proxy}}^{(T-1)}\bigr)
\]

Head computes \(g_{\mathrm{proxy}}^{(T)}=W_{\mathrm{head}}^\top g_y^{(T)}\) for next step. First sample seeds live. **Not EMA.** Stale sign on XOR.

---

### Linear — `TweenSplitLinear`

Affine chain, **skip \(\odot\mathrm{act}'\)**. Walk **matvecs** reverse through the tree. Hemispheres **share** the same down-going vector (siblings, not a product of each other):

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

Same \(O(N^2)\) class as backprop. Score stays StepBP-class unless Acc is way up, because you still pay every \(W^\top\) **and** every `dW`.

---

### LinearCache — `TweenSplitLinearCache` — **dead**

Every 20 steps: full Linear walk, cache per-leaf \(g_i^{\downarrow}\). In between:

\[
g_i\leftarrow g_i^{\mathrm{cache}}\cdot\frac{\|g_y\|_{\mathrm{live}}}{\|g_y\|_{\mathrm{cache}}}
\]

Norm scaling cannot recover sign flips after a frequency switch. Dense sine → ~0–8% Adapt.

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

Other leaves: \(\mathrm{d}W=0\) this sample. Real FLOP cut → Avail 40–50% → Lucy Score explodes. That is a **duty clock**, not a smaller big-O than backprop and not a better chain rule.

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

**Who is “backprop”:** StepBP / TweenChain.  
**Who injects one \(W^\top\):** HeadProxy (with act′), FastProxy (without).  
**Who walks every \(W^\top\):** Linear.  
**Who skips hidden GEMVs on purpose:** Sparse, and hidden HeadProxy/FastProxy `dW`-only.

---

## Cameral (Dense / Bicameral / Tricameral)

The sandwich is always

\[
x \xrightarrow{\text{Dense stem}} h_0 \xrightarrow{\text{mid}} h \xrightarrow{\text{Dense head}} y
\]

**Cameral count** is how the mid is built (`model.go` `buildNativeCameral`):

| `-cam-min` / n | Arch name | Mid |
|----------------|-----------|-----|
| 1 | **Dense** | one Op of that layer kind |
| 2 | **Bicameral** | `HemispheresFrom(n=2, CombineAdd)` |
| 3 | **Tricameral** | `HemispheresFrom(n=3, CombineAdd)` |

“Dense” in the arch column means **one mid Op**, not “the mid is a Dense layer”. A job can be `lstm/Dense` (one LSTM mid) or `lstm/Tricameral` (three LSTM hemispheres, added).

Combine is **add**, not concat and not a gate:

\[
h=\sum_{k=1}^{n} h_k
\]

Backward of add copies the same gap onto every branch (no extra Jacobian):

\[
g_{h_k}=g_h \qquad k=1\ldots n
\]

Credit then does whatever that **mode** does on each leaf. Extra hemispheres add params and a merge. They are **not** automatically more Acc on 2s toys.

Spatial / seq mids (cnn*, convt*, mha, lstm, rnn, mamba, gdn) are View-wrapped so Parallel still sees `[1, hidden]` from the Dense stem:

\[
[1,H] \xrightarrow{\mathrm{View}} \text{rank-}r \xrightarrow{\mathrm{Op}} \text{rank-}r \xrightarrow{\mathrm{View}} [1,H]
\]

`cnn1` / `convt1` copy sits on a **View-wrap floor** (~47.7 Acc / ~50 Soft) for **every** mode including StepBP. That is geometry, not credit.

---

## This sweep

**97,920 jobs**, 2 s, 12 workers, 0 errors.

\[
3\ \mathrm{tasks}\times 34\ \mathrm{dtypes}\times 20\ \mathrm{layers}\times 3\ \mathrm{cams}\times 16\ \mathrm{modes}=97\,920
\]

251 collapsed (Acc < 16 + Cons ≥ 70 + Stab ≥ 90). 3,162 copy-floor rows (`cnn1`/`convt1`).

That is **not** the board the first sentence asks about. Lucy-honest Score is `-duration 10s -workers 1` on Dense sine, float32. Here Score/Avail are shared-CPU. Acc on float32 Dense sine is often already saturated. XOR is a 2 s lottery. Low-bit stores (fp4, int2, binary, …) drag pooled means toward chance. Read **Acc** and **Score** as two different questions.

---

## What Lucy is measuring (and why)

Same formulas as test41 / tide / live_mnist (`lucy/score.go`). Two questions, never one number:

1. **Did it learn?** Acc, SoftAcc, AdaptPct.
2. **What did the duty clock cost?** Avail, Tput, Score.

Sparse winning Score while losing Acc is skipping GEMVs, not a better chain rule. Rivaling backprop is **matched AccΔ vs StepBP**. ScoreΔ is the clock.

Duty times are **thread CPU** on Linux (`RUSAGE_THREAD` in `workclock.go`), not wall. Concurrent workers still steal cache/core, so Avail/Score need `-workers 1` to be Lucy-honest.

### Hard Acc — held-out after the clock

Eval forward on a frozen pool **after** the timed loop. Not inside Score.

| Task | Sample is “correct” when |
|------|--------------------------|
| xor / copy | each dim: \(\mathbf{1}[y_i\ge 0.5]=\mathbf{1}[t_i\ge 0.5]\) (bit match) |
| sine | **every** dim \(\lvert y_i-t_i\rvert < 0.15\) (all-or-nothing 0 or 100) |

\[
\mathrm{Acc}=100\times\frac{\#\text{correct samples}}{\#\text{eval samples}}
\]

XOR Acc clumps at 50/75/100 — a 2-bit pattern in a 16-d vector. Do not rival BP from XOR Acc. Sine Acc can sit at 100% while live SoftAcc is 40% (it fit the *last* frequency, not the jump).

### SoftAcc — live fit during the window

Continuous, every infer, **before** the train step. This is the Acc term **inside Score** (hard Acc is not).

\[
\mathrm{SoftAcc}(y,t)=100\times\Bigl(1-\frac{\lvert y-t\rvert}{s}\Bigr)\quad\text{clamped }[0,100]
\]

Mean over dims, then over samples in the 50 ms pulse.

| Task | helper | scale \(s\) | why that scale |
|------|--------|-------------|----------------|
| sine | `SoftAccOne` | \(0.10\) | Lucy/test41 sine; \(\lvert\mathrm{err}\rvert=0.10\) → SoftAcc 0. The 0.15 hard gate sits just outside that. |
| xor / copy | `SoftAccProb` | \(1.0\) | bits as probabilities vs 0/1; SoftAcc \(\approx 100\times(1-\lvert y-t\rvert)\). |

Why not hard Acc in Score: a 50/75/100 lottery would make the clock jump with one unlucky XOR eval. SoftAcc is the live fit the duty cycle actually paid for.

### Throughput

\[
\mathrm{Tput}=\frac{\text{infer+train samples}}{\text{wall seconds}}
\]

How many sandwich steps fit in the perm. Cheap updates raise Tput; that is not “smarter \(g\)”.

### Availability — infer share of busy time

\[
\mathrm{Avail}=100\times\frac{\mathrm{InferMs}}{\mathrm{InferMs}+\mathrm{TrainMs}}
\]

Why: a credit rule that skips hidden \(W^\top\) / `dW` spends less time in train. Avail is the **duty-cycle** of that skip. BP sits ~12% here because the reverse pass *is* the train. Sparse sits ~40–50% because most leaves get \(\mathrm{d}W=0\). TweenAlt sits ~9% because of the extra re-forward. **This is why Sparse wins every Score table.**

Shared CPU (`-workers` = NumCPU) inflates wait and deflates Avail. Lucy-honest Avail = `-workers 1`.

### Score — the Lucy clock, not the learner

\[
\mathrm{Score}=\frac{\mathrm{Tput}\times\mathrm{Avail}\times\mathrm{SoftAcc}}{10\,000}
\]

The \(10\,000\) is a scale so a ~100 Tput × ~100 Avail × ~100 SoftAcc job lands near \(100\), not \(10^6\). Product on purpose: a mode that is fast and idle-in-train **and** still fitting live outranks a slow perfect fit **on this axis**. That is a mobile/mid-stream ranking (tide / live_mnist), **not** a Jacobian ranking.

If SoftAcc collapsed, Score should die. If SoftAcc held while Avail exploded, Score explodes — read Acc before you quote Score.

### AdaptPct — only sine

Freq pool `1→2→3→4` every `-switch` (default duration/4). A window with `PhaseSwitches>0` is a jump. AdaptPct = mean SoftAcc of the next `-adapt-windows` pulses (default 4; 10s race uses 10).

\[
\mathrm{AdaptPct}=\mathrm{mean}\{\mathrm{SoftAcc}_{i},\ldots,\mathrm{SoftAcc}_{i+K-1}:\text{window }i\text{ switched}\}
\]

xor/copy never switch → AdaptPct \(=0\). LinearCache dies here: cache + \(\lVert g_y\rVert\) scale cannot recover a sign flip after a freq jump.

### The rest (same Lucy snapshot)

\[
\mathrm{ZeroDowntime}=\frac{\mathrm{SoftAcc}\times\mathrm{Avail}}{100}
\qquad
\mathrm{MobileScore}=\frac{\mathrm{Score}}{\mathrm{WeightMiB}}
\]

ZeroDowntime: live fit **while** still inferring (no train-blocked hole). MobileScore: clock per stored weight — low-bit dtypes can win this without winning Acc.

\[
\mathrm{Stability}=\max\bigl(0,\;100-\mathrm{stdev}(\mathrm{SoftAcc}_{\text{windows}})\bigr)
\qquad
\mathrm{Consistency}=100\times\frac{\#\{\text{windows with SoftAcc}\ge 10\}}{\#\text{windows}}
\]

A **collapsed** job in the report is Acc \(<16\) **and** Cons \(\ge 70\) **and** Stab \(\ge 90\): a flat dead net that looks “stable” because nothing moved. 13% Acc + high Cons/Stab is Tween/LinearCache, not a quiet winner.

Time-to-Acc 25/50 and Acc/sec exist on the snapshot; the 2s perm often hits the gate at \(t=0\) or never, so those columns are weak on this file.

---

## Rivaling backprop (matched cells, all 34 dtypes)

2,040 matched cells per mode per task (34 × 20 × 3). Win = AccΔ > 0.5 (ties at 100% do not count).

| Mode | xor AccΔ (win) | sine AccΔ (win) | copy AccΔ (win) | Score win |
|------|----------------:|----------------:|----------------:|-----------|
| TweenChain | +0.0 (13%) | +0.1 (20%) | +0.0 (42%) | ~45% (noise) |
| Tween | −8.7 (10%) | **−21.2** (10%) | −2.7 (32%) | mixed |
| TweenSplit | −1.5 (16%) | −7.5 (29%) | +0.6 (45%) | ~98% |
| HeadProxy | −0.6 (12%) | −3.2 (18%) | −0.0 (40%) | ~100% |
| Linear | −0.5 (13%) | +0.0 (20%) | +0.2 (43%) | ~96% |
| **FastProxy** | **+0.9 (14%)** | **+3.3 (29%)** | **+1.8 (52%)** | ~100% |
| Sparse | **−3.3 (14%)** | **+6.3 (35%)** | −0.0 (43%) | **100%** |
| ProxyAsync | −11.3 (11%) | −0.1 (25%) | −0.8 (36%) | ~99% |
| LinearCache | −18.3 (8%) | −25.4 (11%) | −6.0 (24%) | high Score, dead Acc |
| TweenAlt | −0.2 (18%) | −6.0 (34%) | +0.8 (46%) | **Score loss** (Avail ~9%) |

Allowed claim: FastProxy can **match or beat StepBP Acc** on copy (52% of cells) and on unsaturated sine layers. Sparse **wins the Lucy clock everywhere** and **does not** win XOR Acc; its sine AccΔ is real on some layers/dtypes (lstm, swiglu, gdn, int8/uint8/bfloat16) and a tie/loss on layernorm/rmsnorm/softmax.

Forbidden claim: Sparse is better backprop. ScoreΔ +2640 on XOR is Avail ~50% vs BP ~13%.

Step\* vs non-Step mean Acc is the same family (sine StepBP 48.7 vs NormalBP 48.6). Cellwise \|AccΔ\| of a few points is XOR lottery + 2 s step-count, not a new Jacobian.

---

## Float32 sine — the storage that can rival BP

Pooled-over-dtype sine Acc (~49% StepBP) is the low-bit floor talking. Pin **float32**:

| Mode | mean Acc | Soft | Avail | Adapt | Score |
|------|---------:|-----:|------:|------:|------:|
| StepBP / TweenChain | ~83% | ~37% | ~12% | ~36% | ~290 |
| FastProxy | **88%** | 43% | 27% | 42% | 1165 |
| Sparse | 85% | **47%** | **49%** | **45%** | **3555** |
| HeadProxy | 85% | 35% | 28% | 33% | 1095 |
| Linear | 82% | 39% | 20% | 38% | 712 |
| TweenSplit | ~57% | 25% | 27% | 23% | 560 |
| Tween | 39% | 14% | 15% | 14% | 164 |
| LinearCache | **33%** | 8% | 27% | **8%** | 245 |
| TweenAlt | ~58% | 25% | **9%** | 22% | 65 |
| ProxyAsync | 81% | 37% | 27% | 36% | 1035 |

Dense sandwich only (one mid Op, float32 sine, 20 layers × 16 modes = 320 jobs): StepBP Acc 86%, FastProxy **89%**, Sparse 80%, Tween 36%, LinearCache 38%. SoftAcc: FastProxy 48% vs StepBP 43% vs Sparse 46%. Sparse Score 3215 vs StepBP 533 is still Avail 40% vs 16%.

---

## XOR — the lottery

2-bit XOR in a 16-d vector, threshold 0.5. Acc clumps at **50 / 75 / 100**. All-dtype means: StepBP 59%, FastProxy 59%, Sparse **55%**, Tween 50%, LinearCache 40%, ProxyAsync **47%**.

FastProxy Acc-beats StepBP on **14%** of cells. Sparse Acc-beats on **14%** and **loses mean Acc (−3.3)**. Sparse still **Score-wins 100%** because Avail is ~50% vs BP ~13%. Domain Score winner is `xor/float32/layernorm/Bicameral/Sparse` **15339** at 100% Acc — a duty-clock spike, not a solver.

ProxyAsync is dead on XOR (mean Acc 47%): last sample’s proxy is the wrong sign for a discrete flip.

---

## Copy — Acc and Score finally disagree

Random 16-bit identity. AdaptPct = 0.

- **FastProxy Acc-beats StepBP on 52% of cells** (mean +1.8 Acc). Cleanest “proxy ≥ chain on this toy” once dtypes are in the pool (float32-only used to read +6 / 72% because low-bit stores were absent).
- **Sparse Acc-tied (Δ −0.0)** while Score-winning **every** cell (Δ +1262).
- HeadProxy Acc ~tied with BP; Linear Acc ~tied, Score halfway (still walks every \(W^\top\)).
- **cnn1 and convt1 are a chance floor for every mode including StepBP:** Acc **47.7%**, Soft **49.9%**, 3,162 jobs. `softmax` is the same neighborhood. Those rows are not in the race.

---

## Layers (sine Acc, all dtypes pooled)

| Layer | StepBP | FastProxy | Sparse | Tween | LinearCache |
|-------|-------:|----------:|-------:|------:|------------:|
| gdn | 90.6 | **97.6** | 94.6 | 20.5 | 32.1 |
| rmsnorm | **79.1** | 75.5 | 73.6 | 70.5 | 26.2 |
| layernorm | **77.7** | 75.4 | 73.6 | 70.6 | 27.4 |
| residual | 59.4 | 59.0 | **62.3** | 36.0 | 25.8 |
| rnn | 56.0 | 58.8 | **64.0** | 25.1 | 22.4 |
| dense | 54.3 | 54.5 | **58.2** | 22.7 | 23.7 |
| lstm | 29.7 | 49.7 | **57.9** | 20.0 | 20.2 |
| swiglu | 20.3 | 32.2 | **41.1** | 20.4 | 20.3 |
| kmeans | 21.9 | **35.7** | 35.7 | 20.3 | 20.6 |
| cnn1 / convt1 | ~20 | ~20 | ~20 | ~20 | ~20 |
| softmax | **33.8** | 28.6 | 28.3 | 20.4 | 21.8 |

FastProxy / Sparse **clear** the Split/Tween ceiling on lstm / swiglu / gdn / kmeans. They do **not** beat StepBP on layernorm / rmsnorm / softmax. cnn1/convt1 sine ~20 Acc is the same View-wrap dead zone as the copy floor.

---

## Dtypes (sine Acc)

Weight storage only; activations stay f32. SIMD is Dense+float32; else CPU tiled.

- **float64 / float32 / int64 / complex\***: StepBP ~80–83, FastProxy **86–88**, Sparse ~82–86. This is the rival-BP band.
- **float16**: FastProxy 85, Sparse 86, StepBP 76.
- **int8 / uint8 / bfloat16**: Sparse holds (79 / 79 / 67) while StepBP drops (52 / 50 / 32). That **is** the +6.3 pooled sine AccΔ. It is robustness to coarse storage, not a better \(J^\top\).
- **fp4 / binary / int2 / ternary**: everyone ~20–27. No mode learns.

GDN `SetDType` is a no-op on the mid; stem/head still convert.

---

## Cameral results

Mean Acc by arch is flat:

| Task | Dense (n=1) | Bicameral | Tricameral |
|------|------------:|----------:|-----------:|
| xor | 54.3 | 55.6 | 55.2 |
| sine | 43.0 | 43.5 | 43.2 |
| copy | 54.0 | 54.0 | 54.1 |

Sine headline modes, same story: FastProxy 52.7 / 52.4 / 50.9, Sparse 53.8 / 55.1 / 56.2, StepBP 48.6 / 48.6 / 48.9. Extra hemispheres are extra \(W\) and an add. They do not unlock a new credit rule. Score can rise on Sparse+Tri because there are more leaves to skip. That is still the duty clock.

Linear’s stem gap **sums** hemisphere \(W^\top\) (siblings share \(g_{\mathrm{hemi}}^{\downarrow}\)). Concat cameral was not in this sweep (`CombineAdd` only).

---

## Modes, mapped onto this file

- **StepBP / NormalBP / TweenChain** — real chain rule. Acc ceiling on toys that can be fit. Avail ~12%. Score floor.
- **Tween** — \(P(g_y)\) broadcast. Sine Acc 28% pooled / 39% float32. Dead as a learner.
- **TweenSplit** — \(1/N\,P(g_y)\). Cheap Acc ceiling. Score up from Avail, not from a better \(g\).
- **FastProxy** — \(W_{\mathrm{head}}^\top g_y\), all `dW`. Clears Split’s Acc ceiling on sine/copy. Best Acc rival to StepBP on float32. Score wins vs BP from Avail ~28%, not from skipping the whole net.
- **Sparse** — head + one rotating leaf. Lucy Score winner on every task. Acc held or up on some sine layers and some low-bit dtypes; **lost on XOR**; tied on copy. Do not write “Sparse is better backprop.”
- **HeadProxy** — full \(J_{\mathrm{head}}^\top\). Acc near BP on float32 sine; slightly under on the full dtype pool (act′ wait). Score still Avail-driven.
- **HeadProxyAsync** — XOR disaster; sine/copy roughly FastProxy-lite. Stale proxy is not EMA.
- **Linear** — Acc ≈ BP, Score halfway (extra \(W^\top\) still paid).
- **LinearCache** — dead on sine; XOR Acc 40%; copy Acc 49%. Stale cache + \(\|g_y\|\) scale.
- **TweenAlt** — Split then Tween. Extra forward → Avail ~9% → Score last.

---

## What this file does **not** answer

The README’s first sentence: *Did FastProxy / Sparse beat StepBP on a 10 s sine sandwich?*

On **this** 2 s / 12-worker board:

- **Acc (learn):** FastProxy can match/beat StepBP on copy and on unsaturated sine layers; Sparse’s Acc edge is concentrated on lstm/swiglu/gdn and on int8/uint8/bfloat16, not a general chain-rule win.
- **Score (clock):** Sparse >> FastProxy >> StepBP. That ranking is Avail, and Avail is **not Lucy-honest** with 12 jobs on one CPU.
- **Cameral:** Bi/Tri ≈ one mid Op on Acc. More hemispheres ≠ more intelligence on these toys.

The comparable command is still:

```bash
go run . -dtypes float32 -layers dense -modes stepbp,fastproxy,sparse -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
```

Until that board, the true statements from test48 are: FastProxy can **match or beat StepBP Acc** on copy and on unsaturated sine layers; Sparse **wins the Lucy clock** and **does not** consistently win Acc; LinearCache and Tween stay dead; cnn1/convt1 copy is a floor, not a mode result; extra camerals are an add-merge, not a backprop killer.
