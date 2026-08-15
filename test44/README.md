# Test 44 — ARC-AGI × native camerals

Few-shot is **not** the protocol. One welvet **v0.95.1** Sandwich net per
architecture trains on **all** ARC-AGI training demos, then we ask:

1. Did it reconstruct the demos? (`FitPix`)
2. Did it solve the training tasks’ held-out test grids? (`TrainSolve`)
3. Does that transfer to the evaluation split? (`EvalSolve`, zero-shot)

Cameral arches use the native `Hemispheres(n)` + `Sandwich` + `TrainStackMSE`
API (same stack as `test41_w_native_cam`). Measuring is `welvet/lucy` — the
same SoftAcc / AdaptPct / Availability / Score / MobileScore formulas as
**tide** and **live_mnist**.

Default: **StepTweenChain**, Dense cells, SIMD, **125ms per demo**, one pass
over all 1302 ARC-AGI-1 training demos, then score 400 train tests + 400 eval
tests. Sweeps Dense + bi / tri / quad / 5-cameral (`go run .`). One *n*-cameral
only: `go run . -only N`. Hemisphere kind sweep (no Dense): leftover args or
`-layers`.

```bash
cd apps/aai/test44
go run .                          # Dense + bi..5-cameral, all tiles
go run . -only 20                 # just 20-cameral Dense (not 2..20)
go run . -only 3 cnn cnn2 cnn3 mha lstm
go run . -only 3 -layers all      # every kind except Dense, tricameral each
go run . -n 20 -item-time 50ms    # short smoke
go run . -set agi2 -camerals 5
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-n` | `0` (all) | cap tasks per split |
| `-camerals` | `5` | max hemispheres (`cam-min`..N plus Dense) |
| `-cam-min` | `2` | first cameral count |
| `-only` | `0` | exactly N hemispheres — no Dense, no sweep |
| `-item-time` | `125ms` | TrainStackMSE budget per demo |
| `-passes` | `1` | cycles over the demo set |
| `-layer` | `dense` | single kind when `-layers` / args are empty |
| `-layers` | empty | comma/space list, or `all` (every kind except dense). leftover args also count (`cnn cnn2 mha…`) |
| `-hidden` | `64` | hidden width |
| `-workers` | NumCPU | concurrent cameral nets |
| `-set` | `agi1` | `agi1` \| `agi2` |

Writes `test44_results.json` (gitignored).

---

## Architecture

Pad every grid to 30×30, colors 0–9 as `c/9`, plus `[H/30, W/30]` → **902-d**
vector. Exact ARC solve still requires size + every cell.

| Arch | Sandwich |
|------|----------|
| Dense | `Dense 902→H → Dense H→H → Dense H→902`  (one sequential mid) |
| *n*-cameral | `Dense 902→H → Hemispheres(n, add) → Dense H→902` |

**A cameral is not another layer.** Depth stays three (stem → merge → head).
`Hemispheres(n)` is *n* **parallel** Ops on the same hidden — each hemisphere
has its own weights and can take its own `TrainMode` (`BranchModes`; test41
Mix). Forward is *n* independent views of the stem, then `CombineAdd` (or
avg / concat / MoE gate). That is a split-cognition merge, not `L → L+1`
in a Sequential.

RAM still grows per extra hemi. Stem/head stay Dense `902↔H` adapters so the
ARC vector encoding does not change. `-layers cnn2` / `mha` / `lstm` / … swaps
the *kind* of each hemisphere via `HemispheresFrom` (spatial/seq Ops get a
zero-weight `View` reshape around the hidden vector). *n* is still how many
brains sit side by side. End of a multi-layer run: one ARC+Lucy table **per
layer**, then a **COMPARE all layers** table with the same columns.

---

## How training / “adapt” works (vs test41)

Same **time-box then the target jumps** idea as `test41_w_native_cam`. Not the
same stream.

| | test41 sine | test44 ARC |
|--|-------------|------------|
| Clock | Infer+train every tick for **10s** | **125ms** `TrainStackMSE` on one pair, **one** forward, next pair |
| What jumps | Sine **frequency** `1x→2x→3x→4x` every 2.5s | The **demo pair** (new grid; new **task file** when the id changes) |
| Weights | One net the whole race | One net the whole 1302-demo pass |
| AdaptPct | SoftAcc in the first windows **after a freq flip** | SoftAcc on the **first 4 demos after a new task id** |
| After the race | Still on sine | Freeze → score training-task **test** grids, then eval **zero-shot** |

So: train on this tile for `item-time`, pulse Lucy, then the mapping is
**replaced** — not a frequency nudge on a living sine. Same Sandwich has to
re-adapt every 125ms. Task-id change is the ARC analog of a test41 switch;
pairs inside one JSON are extra demos of the *same* rule before the next
file.

Then we stop training and ask whether any of that stuck on held-out train
tests and on eval.

---

## What adding camerals actually does

Snapshot, same protocol (`-item-time 125ms -passes 1 -hidden 64`, StepTweenChain,
SIMD, ~2m45–2m50s wall). Dense–5 ran together; 20 and 100 were `-only N`.

| Arch | KiB | FitPix | TrainPix | EvalPix | MeanLoss | SoftAcc | AdaptPct | Steps |
|------|----:|-------:|---------:|--------:|---------:|--------:|---------:|------:|
| Dense (1 mid) | 467 | 11.6% | 10.4% | 8.3% | 0.148 | 67.4% | 67.5% | 76.7k |
| Bicameral | 483 | 13.6% | 12.4% | 8.9% | 0.111 | 74.8% | 74.8% | 75.3k |
| Tricameral | 499 | 12.3% | 10.8% | 8.0% | 0.111 | 74.9% | 74.9% | 71.2k |
| Quadcameral | 515 | **14.3%** | **12.9%** | **9.3%** | 0.103 | 76.6% | 76.6% | 67.6k |
| 5-cameral | 531 | 13.2% | 11.8% | 8.4% | 0.094 | 78.4% | 78.4% | 64.4k |
| 20-cameral | 771 | 13.9% | 12.0% | 8.6% | **0.052** | **87.2%** | **87.5%** | 60.1k |
| **100-cameral** | 2051 | 3.9% | 3.4% | 2.2% | 0.075 | 82.1% | 82.2% | **17.9k** |

**Official ARC solves: 0/400 train, 0/400 eval, every arch.** A padded Dense
mapper is not a program synthesizer. Pixel/SoftAcc are the live signal.

### What *does* move with *n*

1. **SoftAcc climbs into the 20s, then slips at 100** (67% → 78% → **87% at
   20** → 82% at 100). Each extra cameral is another parallel path through
   the same stem, add-merged. AdaptPct tracks SoftAcc (switch = next ARC
   file). On the 20-cameral pass rolling SoftAcc went ~52% → ~97% through
   the epoch. 100-cameral still sits above Dense/5 but **loses to 20**.

2. **Train MSE bottoms at 20, not 100** (0.148 → 0.094 → **0.052 at 20** →
   0.075 at 100). 20-cameral halved 5-cameral’s loss; 100-cameral gave some
   of that back. More brains only help while the 125ms budget still buys
   enough SGD ticks.

3. **Hard pixel peaks at Quad, dies at 100.** Quad still holds best
   Fit/Train/Eval pixel (14.3 / 12.9 / 9.3). 20-cameral (13.9 / 12.0 / 8.6)
   did not beat it. **100-cameral collapsed to 3.9 / 3.4 / 2.2** — worse
   than Dense — and **never hit t50** (t50=0). SoftAcc can stay ~82% while
   rounded 0–9 cells fall apart.

4. **Steps collapse with *n*** (76.7k → 60.1k at 20 → **17.9k at 100**,
   ~14 ticks per 125ms vs ~46 at 20 vs ~59 at Dense). Each tick is *n*
   Dense GEMVs. Wall time stays ~2m50s (`item-time` cadence) but 100-cameral
   is **starved of updates**: 2.0 MiB, infer 1.3s vs 20-cameral’s 0.42s.
   Avail ticked up to 0.8% only because infer got slower, not because we
   served more.

5. **Eval stays below train** on every arch. 100-cameral did not close it
   (3.4 train / 2.2 eval); it just made both worse.

### 20-cameral pulse (same protocol, `-only 20`)

Live log: `item=` is MSE on the **current** demo after 125ms; `soft`/`pix`
are that pulse’s Lucy/pixel. SoftAcc on the rolling 100 went 52 → 80 → 89
→ 97 while **pix stayed noisy 0–12%**. That is the adapt loop working as
designed: the merge can hug the continuous `c/9` canvas after a task
switch; discrete grids still miss. Official solves stayed **0/400 + 0/400**.
Lucy Score still prints 0 (Avail ~0.3%, train 161s vs infer 0.4s). Stab
rose a bit (80% → 84%). Acc/s 0.528 vs 5-cameral 0.475.

### 100-cameral (`-only 100`) — past the cliff

Same 125ms/demo, ~2m50s. Weights **2051 KiB**. Steps **17.9k** (~14/item).
Fit/Train/Eval pixel **3.9 / 3.4 / 2.2**. SoftAcc 82.1 / Adapt 82.2 — still
“high” on scale 1.0 colors, but **below 20-cameral**, and mean loss **0.075**
is worse than 20’s 0.052. t25≈5.9s, **t50 never**. Infer 1.3s (heavier
forward) vs train 164s.

This is the 125ms budget **losing to Parallel width**: 100 add-merged Dense
twins cost so much per tick that the net cannot finish adapting a tile
before the next one arrives. More camerals stopped being more cognition
and became fewer updates.

### What did *not* move

- **Exact grids / TrainSolve / EvalSolve** — still zero. Adding camerals
  does not invent objectness, counting, or crop-to-size. CNN/MHA/LSTM as
  *hemispheres on the 64-d stem* (below) did not change that. This is still
  a continuous interpolator on a 30×30 canvas.
- **Lucy Score and MobileScore printed as 0.** Score =
  `Throughput × Availability × SoftAcc / 10_000`. Availability is
  `InferMs / (InferMs+TrainMs)` ≈ **0.2%** because we spend 125ms training
  and only a fraction of a millisecond on the pulse forward. SoftAcc of 78
  with Avail 0.2 and Tput ~21 still floors Score at 0 in the integer
  column. That is a **duty-cycle artifact of this protocol**, not “5-cameral
  is worthless.” Tide/live_mnist keep Score alive by serving while training;
  test44 is train-heavy by design. Compare **SoftAcc, pixel, loss, steps,
  KiB** until we interleave more infer.

- **Consistency 100%** — every pulse SoftAcc was above Lucy’s 10%
  threshold. Uninformative on this encoding (color SoftAcc sits ~70% even
  when pixel is ~12%, because off-by-one color still scores ~89 on scale
  1.0).

### How to read a cameral delta

| If this goes up with *n* | It means |
|--------------------------|----------|
| SoftAcc / AdaptPct | Better continuous fit of the *current* demo after 125ms |
| FitPix / TrainPix | More rounded cells match on tiles the net has trained |
| EvalPix | A little of that fit survives on unseen tasks (still << solve) |
| MeanLoss | Tighter MSE on the last pulse of each demo |
| KiB | Extra parallel hemi (`H×H` Dense twin in *this* run) |
| Steps ↓ | *n* GEMVs per tick in Parallel — budget buys fewer SGD steps |

| If this stays flat | It means |
|--------------------|----------|
| TrainSolve / EvalSolve | No exact ARC programs yet |
| Score / MobileScore (int) | Availability starved by 125ms-train / 1-forward |
| Throughput | Same demo cadence (`1 / item-time`) for every arch |

**Working read:** Dense → Quad → 20 → 100 is **width at one depth**, not
layers. SoftAcc/loss **peak around 20**; **hard pixel peaks at Quad**;
**100-cameral is a regression** (pixel worse than Dense, fewer steps, 2 MiB).
Exact ARC solves did not appear at any *n*. Heterogeneous hemispheres
(`-layers all`) ran next — they did **not** beat Dense twins on pixel.

---

## Tricameral layer sweep (`-only 3 -layers all`)

Same protocol as the Dense *n*-sweep: **125ms/demo**, one pass, H=64,
StepTweenChain, ARC-AGI-1 1302 demos → 400 train tests + 400 eval.
`go run . -only 3 -layers all` — **19 nets**, 12 workers, wall **5m44s**.
Stem/head stay Dense `902↔64`; only the three parallel hemispheres change
kind. Spatial/seq Ops see a `View` of that hidden vector (e.g. CNN2
`4×4×4`, MHA/LSTM `8×8`), **not** the raw 30×30 grid.

Dense tricameral from the earlier SIMD sweep is the reference (not re-run
in this batch): Fit/Train/Eval **12.3 / 10.8 / 8.0**, loss 0.111, Soft 74.9%,
71.2k steps, 499 KiB.

### TLDR

**Swapping the hemisphere Op did not beat Dense camerals, and did not
solve a single ARC task.** Official solves stayed **0/400 + 0/400**. Best
hard pixel in this sweep is **residual** (10.0 / 9.3 / 8.1) — still under
Dense tri and well under Quad. Residual *is* Dense `F(x)+x`, so the
winner among “other layers” is the one closest to Dense.

Two traps in the Lucy columns:

1. **LayerNorm / RMSNorm look like champions** (SoftAcc **91%**, loss
   **0.023–0.025**) while FitPix is only **~6.5%**. Norms pull the 64-d
   vector toward a stable mean; `c/9` SoftAcc loves that. Rounded 0–9
   cells do not. Do not rank this sweep on SoftAcc alone.
2. **Lucy Score / MobileScore still print 0** (Avail 0.6–5.7%). The
   COMPARE “winner” is `cnn1` only because every Score integer-floors to
   0 and cnn1 is first. Rank **pixel / loss / steps**, not Score.

Dead cluster stuck at Soft **~51.4%**, loss **~0.240**, FitPix **~5%**
(cnn1, cnn3, convt1, convt3, mha, gdn, swiglu, kmeans, softmax): they
barely left the first-100-demo plateau. CNN1/CNN3 only got **~4 SGD ticks
per 125ms** (5.5k / 5.3k steps vs softmax 46k). MHA was also thin (10.8k).
This is the same 125ms-vs-width cliff as 100-cameral, now as **expensive
Ops** instead of 100 Dense twins.

What actually moved a little: **RNN** (loss 0.068, Soft 80%, pix 7.1) —
best continuous fit of a real mixer; **Mamba** best eval pixel among
exotics (**7.3%**); **ConvT2** trained (loss 0.184, Soft 61%) while
ConvT1/3 did not; **metacognition** tracked residual at a discount
(7.0 / 6.3 / 5.2). None of that is an ARC program. The 30×30 grid is
still crushed through a Dense stem before the fancy hemisphere ever sees
it.

### COMPARE (all 19, tricameral)

| Layer | KiB | FitPix | TrainPix | EvalPix | MeanLoss | SoftAcc | AdaptPct | Steps |
|-------|----:|-------:|---------:|--------:|---------:|--------:|---------:|------:|
| cnn1 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 5.6k |
| cnn2 | 453 | 5.9% | 5.8% | 4.5% | 0.240 | 51.5% | 51.5% | 14.9k |
| cnn3 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 5.3k |
| convt1 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 38.7k |
| convt2 | 453 | 6.6% | 5.9% | 4.8% | 0.184 | 60.7% | 60.8% | 34.1k |
| convt3 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 35.7k |
| mha | 454 | 5.1% | 5.2% | 4.0% | 0.240 | 51.4% | 51.4% | 10.8k |
| lstm | 458 | 6.6% | 5.3% | 4.4% | 0.221 | 53.7% | 53.6% | 21.9k |
| **rnn** | 453 | 7.1% | 6.8% | 5.2% | **0.068** | 80.2% | 80.2% | 38.0k |
| mamba | 456 | 8.3% | **8.4%** | **7.3%** | 0.213 | 55.6% | 55.7% | 15.1k |
| gdn | 451 | 5.2% | 5.5% | 4.2% | 0.240 | 51.4% | 51.4% | 34.7k |
| swiglu | 847 | 5.2% | 5.4% | 4.5% | 0.241 | 51.4% | 51.4% | 15.1k |
| **residual** | 499 | **10.0%** | **9.3%** | **8.1%** | 0.125 | 70.5% | 70.7% | 24.1k |
| sequential | 547 | 6.5% | 6.5% | 5.2% | 0.213 | 55.4% | 55.5% | 26.9k |
| softmax | 451 | 5.5% | 5.1% | 4.5% | 0.238 | 51.6% | 51.6% | 45.8k |
| layernorm | 452 | 6.7% | 6.4% | 5.6% | **0.025** | **91.4%** | **91.4%** | 37.8k |
| rmsnorm | 452 | 6.4% | 6.2% | 4.7% | **0.023** | **91.7%** | **91.7%** | 45.7k |
| kmeans | 499 | 5.4% | 5.4% | 4.4% | 0.240 | 51.4% | 51.4% | 33.0k |
| metacognition | 499 | 7.0% | 6.3% | 5.2% | 0.146 | 66.3% | 66.4% | 27.1k |
| *Dense tri (prior run)* | 499 | *12.3%* | *10.8%* | *8.0%* | *0.111* | *74.9%* | *74.9%* | *71.2k* |

Official ARC: **0 exact / 0 solved** on every row. Consistency 100% on every
row (SoftAcc never dropped under Lucy’s 10% floor — uninformative).

### Observations

1. **Kind ≠ vision on the grid.** CNN/MHA never saw 30×30. They mixed a
   Dense stem’s 64-d code. A “CNN cameral” here is not a convnet ARC
   solver; it is three tiny convs on a reshaped hidden. That is why cnn1
   and Dense-quad are not comparable as “CNN vs Dense.”

2. **Steps are the hidden axis.** cnn1/cnn3 ~4 ticks/item; rmsnorm/softmax
   ~35. Same 125ms. Ops that im2col or attn a 64-d view still cost enough
   that the net never leaves the ~0.24 MSE basin. ConvT1 got 38k steps and
   *still* sat at 0.241 — width isn’t the only failure; some Ops just
   don’t move this encoding.

3. **Residual ≈ Dense with a skip.** 499 KiB (same as Dense tri), pixel
   closest to the Dense reference, Soft 70.5 vs Dense 74.9. The skip helps
   train (loss 0.125) but does not invent structure. Sequential (two Dense
   in the hemi) was worse pixel than residual and slower to fit.

4. **RNN vs LSTM.** Vanilla RNN (38k steps, loss 0.068, Soft 80) beat LSTM
   (22k, 0.221, 53.7) on every useful column. LSTM is heavier per tick on
   seq=8, d=8; the budget prefers the cheap recurrence. Pixel still 7%.

5. **Mamba** is the only exotic with EvalPix **7.3%** (train 8.4, fit 8.3)
   — small lift over the dead cluster, still below residual and Dense.
   GDN looked like the dead cluster despite 35k steps (linear-attn on
   seq=8 is not ARC).

6. **SwiGLU** was the RAM hog (847 KiB) and still dead metrics. Extra FFN
   width in the hemi does not help a 902-d color canvas.

7. **Softmax / KMeans / GDN / MHA** as hemispheres are close to a no-op
   on this protocol: SoftAcc glued to 51.4% (the “always ~0.5” color
   guess). MHA with seq=8, d=8, 4 heads is a toy attention, not a
   transformer on the grid.

8. **Lucy Score ranking is noise.** Avail 0.6–5.7% (cnn1/cnn3 look
   “available” only because they were *slow to infer*, 2.1–2.3s infer vs
   residual 0.7s). Integer Score = 0 for all 19.

**Working read:** keep Dense camerals for pixel; use this sweep as a
negative — **hemisphere *kind* on a 64-d stem is not the ARC lever**. Width
on Dense-like residuals is a different story (`-only 15` below). Next
deltas that could actually change the encoding: CNN/MHA **on the 30×30
grid** (no Dense flatten first), a discrete size/color head, or Mix
`BranchModes`. Not “try cnn3 again at n=20” on the same 902-d sandwich.

---

## 15-cameral layer sweep (`-only 15 -layers all`)

Same protocol, **15 hemispheres** per kind. Wall **6m08s**. COMPARE table
from the run; Dense-15 was not in this batch — use Dense-5 / Quad / Dense-20
from the SIMD *n*-sweep as pixel references.

### TLDR

**n=15 does not magically turn CNN/MHA into ARC solvers** (still **0/400**
exact). It **does** scale the one kind that was already Dense-shaped:
**residual 15-cameral is the best eval pixel of any test44 run so far**
(**10.4%** vs Quad **9.3%**, Dense-20 **8.6%**, residual-3 **8.1%**).
TrainPix **12.7%** ties Quad. Fit **13.5%** is still a hair under Quad’s
14.3. Residual at n=15 is “Dense twins with a skip,” not a new inductive
bias — and it **beats Dense-20 on eval** at 691 KiB vs 771 KiB.

Everything expensive got **starved**. MHA 3.5k steps, Mamba 4.2k, SwiGLU
4.4k at **2431 KiB**, sequential 5.2k. Mamba’s n=3 eval lift (**7.3%**)
**vanished** (4.2%). CNN1/3, ConvT1/3, GDN, KMeans still glued at Soft
51.4 / loss 0.24. Lucy “winner” is **mha** with Score **1** only because
infer took **4.4s** (Avail 8.3%, Tput 14) — worst duty cycle, not best
mapper.

### n=3 → n=15 (movers vs dead)

| Layer | Fit 3→15 | Train 3→15 | Eval 3→15 | Loss 3→15 | Soft 3→15 | Steps 3→15 |
|-------|----------|------------|-----------|-----------|-----------|------------|
| **residual** | 10.0→**13.5** | 9.3→**12.7** | 8.1→**10.4** | 0.125→**0.087** | 70.5→**79.1** | 24.1k→16.6k |
| metacognition | 7.0→**10.5** | 6.3→**10.0** | 5.2→**8.3** | 0.146→0.112 | 66.3→73.7 | 27.1k→10.8k |
| layernorm | 6.7→**10.3** | 6.4→8.7 | 5.6→6.9 | 0.025→0.086 | 91.4→89.7 | 37.8k→29.7k |
| rnn | 7.1→8.1 | 6.8→7.3 | 5.2→5.6 | 0.068→0.077 | 80.2→80.4 | 38.0k→16.6k |
| cnn2 | 5.9→7.3 | 5.8→6.2 | 4.5→5.0 | 0.240→0.206 | 51.5→56.6 | 14.9k→12.3k |
| **mamba** | 8.3→**5.2** | 8.4→**5.3** | 7.3→**4.2** | 0.213→0.239 | 55.6→51.5 | 15.1k→**4.2k** |
| convt2 | 6.6→5.7 | 5.9→5.1 | 4.8→**3.3** | 0.184→0.174 | 60.7→62.5 | 34.1k→20.4k |
| rmsnorm | 6.4→6.0 | 6.2→4.8 | 4.7→**3.5** | 0.023→0.097 | 91.7→88.7 | 45.7k→43.2k |
| sequential | 6.5→5.4 | 6.5→5.7 | 5.2→4.3 | 0.213→0.240 | 55.4→51.4 | 26.9k→**5.2k** |
| swiglu | 5.2→5.3 | — | — | 0.241 | 51.4 | 15.1k→4.4k (**847→2431 KiB**) |

### COMPARE (all 19, 15-cameral)

| Layer | KiB | FitPix | TrainPix | EvalPix | MeanLoss | SoftAcc | AdaptPct | Steps |
|-------|----:|-------:|---------:|--------:|---------:|--------:|---------:|------:|
| cnn1 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 8.1k |
| cnn2 | 459 | 7.3% | 6.2% | 5.0% | 0.206 | 56.6% | 56.7% | 12.3k |
| cnn3 | 453 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 6.6k |
| convt1 | 451 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 29.3k |
| convt2 | 459 | 5.7% | 5.1% | 3.3% | 0.174 | 62.5% | 62.6% | 20.4k |
| convt3 | 453 | 4.8% | 5.1% | 3.8% | 0.241 | 51.4% | 51.4% | 22.0k |
| mha | 466 | 5.4% | 5.3% | 4.6% | 0.241 | 51.4% | 51.4% | **3.5k** |
| lstm | 485 | 5.8% | 5.4% | 4.4% | 0.227 | 53.0% | 53.0% | 7.7k |
| rnn | 459 | 8.1% | 7.3% | 5.6% | 0.077 | 80.4% | 80.3% | 16.6k |
| mamba | 474 | 5.2% | 5.3% | 4.2% | 0.239 | 51.5% | 51.6% | 4.2k |
| gdn | 451 | 5.4% | 5.2% | 4.2% | 0.240 | 51.4% | 51.4% | 13.5k |
| swiglu | **2431** | 5.3% | 5.1% | 4.3% | 0.241 | 51.4% | 51.4% | 4.4k |
| **residual** | 691 | **13.5%** | **12.7%** | **10.4%** | **0.087** | 79.1% | 79.2% | 16.6k |
| sequential | 931 | 5.4% | 5.7% | 4.3% | 0.240 | 51.4% | 51.5% | 5.2k |
| softmax | 451 | 5.8% | 5.4% | 4.7% | 0.212 | 54.5% | 54.5% | 30.2k |
| layernorm | 458 | 10.3% | 8.7% | 6.9% | 0.086 | **89.7%** | **89.8%** | 29.7k |
| rmsnorm | 455 | 6.0% | 4.8% | 3.5% | 0.097 | 88.7% | 88.7% | 43.2k |
| kmeans | 691 | 5.4% | 5.3% | 4.5% | 0.240 | 51.4% | 51.5% | 15.3k |
| metacognition | 691 | 10.5% | 10.0% | 8.3% | 0.112 | 73.7% | 73.8% | 10.8k |
| *Dense-5 (prior)* | *531* | *13.2%* | *11.8%* | *8.4%* | *0.094* | *78.4%* | *78.4%* | *64.4k* |
| *Quad (prior)* | *515* | *14.3%* | *12.9%* | *9.3%* | *0.103* | *76.6%* | *76.6%* | *67.6k* |
| *Dense-20 (prior)* | *771* | *13.9%* | *12.0%* | *8.6%* | *0.052* | *87.2%* | *87.5%* | *60.1k* |

Official ARC: **0 exact / 0 solved** on every row. Consistency 100%.

### Observations

1. **Residual width is the Dense-cameral story again.** 3→15: Fit +3.5,
   Train +3.4, Eval **+2.3**, loss 0.125→0.087, Soft 70→79, at the usual
   step tax (24k→17k). Eval **10.4%** is the first time a non-Dense-twin
   sandwich beats Quad/Dense-20 on *transfer pixel*. Still not a solve.
   691 KiB sits between Dense-5 (531) and Dense-20 (771).

2. **Metacognition tracks residual** (it wraps a Dense) but starved
   harder (10.8k steps). 10.5 / 10.0 / 8.3 ≈ Dense-5 pixel, t50 **97s**
   (late). Heuristic wrapper + 15 copies is extra cost, not extra ARC.

3. **LayerNorm n=15 started mapping cells** (Fit 6.7→10.3) while SoftAcc
   stayed ~90 and **loss got worse** (0.025→0.086). The n=3 “91% Soft /
   6% pix” cheat loosened: more parallel γ/β actually moved rounded
   colors. RMSNorm did the opposite (Eval 4.7→3.5, loss 0.023→0.097) —
   do not treat the two norms as interchangeable.

4. **RNN plateaued.** Soft stayed ~80, pixel +1, loss slightly worse,
   steps halved. Cheap recurrence saturates; more RNNs ≠ more structure.
   LSTM still dead-ish (7.7k steps).

5. **Mamba / sequential / SwiGLU / MHA hit the 100-cameral cliff** at
   n=15 because each hemi is fat. Mamba lost its only interesting column.
   SwiGLU **2.4 MiB** for 5.3% Fit is the RAM punchline. Sequential
   (two Dense per hemi × 15) collapsed back to the 51.4% Soft basin.

6. **CNN2 nudged** (5.9→7.3 Fit, loss 0.24→0.21, Soft 51→57). CNN1/3 did
   not (still ~4–5 ticks/item). Tiny convs on a 64-d view still aren’t
   vision; extra width only helps the 2d layout a little.

7. **ConvT2 pixel went down** (Eval 4.8→3.3) even though loss improved
   slightly. More transpose-conv copies overfit the continuous canvas
   and hurt discrete cells — same SoftAcc-vs-pixel split as Dense-100.

8. **Score “winner” mha is Availability theater.** Infer 4.4s / train
   48s → Avail 8.3% → integer Score 1. Soft 51.4, pix ~5%, **3.5k steps**.
   cnn1/cnn3 also look “available” (infer 4–5s). Rank pixel, not Score.

**Working read:** *n* still only helps **Dense-like** hemispheres
(residual, a bit of metacognition / layernorm). Fat kinds die at 15 the
way Dense twins died at 100. Residual-15’s eval edge over Quad is the
one positive delta in this file that is not a SoftAcc illusion — and it
is still a 10% color canvas, **0 ARC solves**. Next: grid-native CNN/MHA,
not n=50 residual.

---

## Lucy / tide formulas (unchanged)

```
SoftAcc      = 100 × (1 − |pred−target| / scale)   clamped [0,100]
             // colors use scale 1.0 on c/9  (same idea as live_mnist p(true))
Availability = InferMs / (InferMs + TrainMs) × 100
AdaptPct     = mean SoftAcc on the first 4 demos after each task-id switch
Throughput   = TotalOutputs / duration_seconds
Score        = Throughput × Availability × SoftAcc / 10_000
ZeroDowntime = SoftAcc × Availability / 100
MobileScore  = Score / WeightMiB
```

Hard pixel (`AvgAccuracy` / FitPix) is recorded separately. Score uses SoftAcc.
AdaptPct here switches on **ARC task file**, not a sine frequency or MNIST
label flip — recovery windows are “next tile,” not “mid-stream phase B.”
