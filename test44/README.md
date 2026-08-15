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
tests. Sweeps Dense + bi / tri / quad / 5-cameral.

```bash
cd apps/aai/test44
go run .                          # full train → train-solve → eval
go run . -n 20 -item-time 50ms    # short smoke
go run . -set agi2 -camerals 5
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-n` | `0` (all) | cap tasks per split |
| `-camerals` | `5` | max hemispheres (`cam-min`..N plus Dense) |
| `-cam-min` | `2` | first cameral count |
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
| Dense | `Dense 902→H → Dense H→H → Dense H→902` |
| *n*-cameral | `Dense 902→H → Hemispheres(n, add) → Dense H→902` |

Hemispheres are **n Dense twins**, merged with `CombineAdd`. Extra *n* only
adds `n × H × H` weights in the middle — stem and head stay shared. That is
why RAM only grows ~16 KiB per extra hemi at `H=64` (Dense 467 KiB →
5-cameral 531 KiB).

Later `-layer cnn2` / `mha` / … will swap those twins via `HemispheresFrom`
without changing the ARC loop.

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
| 5-cameral | 531 | 13.2% | 11.8% | 8.4% | **0.094** | **78.4%** | **78.4%** | 64.4k |

**Official ARC solves: 0/400 train, 0/400 eval, every arch.** A padded Dense
mapper is not a program synthesizer. Pixel/SoftAcc are the live signal.

### What *does* move with *n*

1. **SoftAcc climbs with hemisphere count** (67% → 78%). That is the Lucy
   color-vector score (scale 1.0 on `c/9`), measured after each demo’s 125ms
   train pulse. More parallel Dense twins, add-merged, fit the *current*
   padded grid more tightly. AdaptPct tracks SoftAcc because “task switch”
   here is just the next ARC file — the first four demos of a new tile look
   like the rest of that pulse, not a sine-frequency shock.

2. **Train MSE falls with *n*** (0.148 → 0.094). Same story: extra
   hemispheres are extra capacity on the merge, so last-item loss after the
   budget is lower. This is **not** the same as solving the task — the net
   is overfitting one 902-d pair at a time.

3. **Pixel is weakly better than Dense, not strictly monotonic.** Quad
   posted the best Fit/Train/Eval pixel; Tri dipped below Bi; 5-cameral sat
   between Bi and Quad on pixel while winning SoftAcc. Add-merge of *n*
   identical Dense twins is **not** “n times smarter at ARC.” It is a
   smoother continuous fit (SoftAcc/loss) that only sometimes lines up with
   hard cell hits after rounding to 0–9.

4. **Steps per wall-second fall as *n* grows** (76.7k → 64.4k). Each
   `TrainStackMSE` runs *n* Dense GEMVs in the hemi. The 125ms budget buys
   fewer updates on 5-cameral than on Dense. That is the cost of the extra
   SoftAcc: **heavier ticks, not free lunch.**

5. **Eval pixel stays below train pixel** (~2–4 points) on every arch.
   Zero-shot eval is a *different set of transformations*. Extra camerals
   did not close that gap; they slightly lifted both sides. Generalization
   here is “a bit more of the same padded mapping,” not a learned rule.

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
| KiB | Extra `H×H` hemi (linear in *n*) |
| Steps ↓ | Each tick is more GEMV — budget buys fewer SGD steps |

| If this stays flat | It means |
|--------------------|----------|
| TrainSolve / EvalSolve | No exact ARC programs yet |
| Score / MobileScore (int) | Availability starved by 125ms-train / 1-forward |
| Throughput | Same demo cadence (`1 / item-time`) for every arch |

**Working read of this run:** going Dense → 5-cameral is a **capacity /
smoothness** knob on the native Parallel merge, not an ARC solver. Quad
was the sweet spot for **hard pixel** on this matrix; 5-cameral won
**SoftAcc and loss** while paying more RAM and fewer steps. Next
interesting delta is swapping Dense twins for other `-layer` kinds, not
stacking more identical hemispheres.

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
