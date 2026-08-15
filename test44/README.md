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

Snapshot from the first full ARC-AGI-1 run
(`-item-time 125ms -passes 1 -hidden 64`, StepTweenChain, SIMD, ~2m45s wall
with 5 nets in parallel):

| Arch | KiB | FitPix | TrainPix | EvalPix | MeanLoss | SoftAcc | AdaptPct | Steps |
|------|----:|-------:|---------:|--------:|---------:|--------:|---------:|------:|
| Dense | 467 | 11.6% | 10.4% | 8.3% | 0.148 | 67.4% | 67.5% | 76.7k |
| Bicameral | 483 | 13.6% | 12.4% | 8.9% | 0.111 | 74.8% | 74.8% | 75.3k |
| Tricameral | 499 | 12.3% | 10.8% | 8.0% | 0.111 | 74.9% | 74.9% | 71.2k |
| Quadcameral | 515 | **14.3%** | **12.9%** | **9.3%** | 0.103 | 76.6% | 76.6% | 67.6k |
| 5-cameral | 531 | 13.2% | 11.8% | 8.4% | 0.094 | 78.4% | 78.4% | 64.4k |
| **20-cameral** (`-only 20`) | 771 | 13.9% | 12.0% | 8.6% | **0.052** | **87.2%** | **87.5%** | 60.1k |

**Official ARC solves: 0/400 train, 0/400 eval, every arch.** A padded Dense
mapper is not a program synthesizer. Pixel/SoftAcc are the live signal.

### What *does* move with *n*

1. **SoftAcc climbs with hemisphere count** (67% → 78% → **87% at 20**).
   Lucy color-vector score after each demo’s 125ms pulse. Each extra cameral
   is another parallel path through the same stem, add-merged — more
   independent fits of the *current* grid, not a deeper stack. AdaptPct
   tracks SoftAcc because the “switch” is the next ARC file, not a sine
   frequency shock. On the 20-cameral pass the rolling SoftAcc itself
   climbed through the epoch (~52% at demo 100 → ~97% by demo 1300): the
   same net is getting better at slamming each new pair in 125ms as it
   walks the set.

2. **Train MSE falls with *n*** (0.148 → 0.094 → **0.052 at 20**). More
   hemispheres in the merge, each updated on its own branch. Still not
   “solved the task” — one 902-d pair at a time. 20-cameral roughly **halved
   5-cameral’s mean loss** while FitPix only moved 13.2% → 13.9%.

3. **Pixel is weakly better than Dense, not strictly monotonic.** Quad
   still holds **best Fit/Train/Eval pixel** (14.3 / 12.9 / 9.3). 20-cameral
   (13.9 / 12.0 / 8.6) sits next to 5-cameral, **below Quad**. Extra
   viewpoints keep buying SoftAcc/loss; rounded 0–9 cells do not keep
   pace. *n* brains ≠ *n*× ARC, and ≠ stacking another sequential layer.

4. **Steps per wall-second fall as *n* grows** (76.7k → 64.4k → **60.1k**
   at 20; ~50 SGD ticks per 125ms vs Dense ~59). Each `TrainStackMSE` runs
   *n* Dense GEMVs in the hemi. 20-cameral still finished in **~2m45s**
   (same `item-time` cadence) but each tick is heavier: 771 KiB vs Dense
   467. SoftAcc is not free.

5. **Eval pixel stays below train pixel** (~2–4 points) on every arch,
   including 20-cameral (12.0 train / 8.6 eval). Zero-shot eval is a
   different set of transformations. Twenty hemispheres did not close that
   gap.

### 20-cameral pulse (same protocol, `-only 20`)

Live log: `item=` is MSE on the **current** demo after 125ms; `soft`/`pix`
are that pulse’s Lucy/pixel. SoftAcc on the rolling 100 went 52 → 80 → 89
→ 97 while **pix stayed noisy 0–12%**. That is the adapt loop working as
designed: the merge can hug the continuous `c/9` canvas after a task
switch; discrete grids still miss. Official solves stayed **0/400 + 0/400**.
Lucy Score still prints 0 (Avail ~0.3%, train 161s vs infer 0.4s). Stab
rose a bit (80% → 84%). Acc/s 0.528 vs 5-cameral 0.475.

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

**Working read:** Dense → 5 → **20** adds **parallel hemispheres at one
depth**, not layers. SoftAcc/loss keep moving with *n* (20-cameral 87% /
0.052); **hard pixel peaked at Quad** and 20-cameral did not beat it.
Exact ARC solves did not appear. Next structural delta is heterogeneous
hemispheres (`-layer` / `HemispheresFrom`) or Mix `BranchModes`, not a
deeper Sequential and not blindly scaling *n* for pixel.

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
