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
only: `go run . -only N`.

```bash
cd apps/aai/test44
go run .                          # Dense + bi..5-cameral, all tiles
go run . -only 20                 # just 20-cameral (not 2..20)
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
| `-layer` | `dense` | stem/hemis/head kind (`cnn2`/`mha`/… reserved) |
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

RAM still grows ~`H×H` per extra hemi at `H=64` (Dense 467 KiB → 5-cameral
531 KiB) because this run used Dense twins. The *topology* is parallel, not
“one more Dense in the stack.” Later `-layer cnn2` / `mha` / … swaps the
*kind* of each hemisphere via `HemispheresFrom`; *n* is still how many
brains sit side by side.

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
  does not invent objectness, counting, or crop-to-size. Until CNN/MHA
  (or a size head that is actually used as a discrete program) land, this
  is a continuous interpolator on a 30×30 canvas.
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
Exact ARC solves did not appear at any *n*. Next delta is heterogeneous
hemispheres (`-layer` / `HemispheresFrom`) or Mix `BranchModes` — not
`n=100` on the same Dense twins.

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
