# AAI — Adaptive Artificial Intelligence

**AAI** is OpenFluke’s lab for **live** intelligence: nets that **serve while they train**, adapt when the world flips, and still fit in a small box.

It is not a chatbot product. It is not the Welvet engine. It is the experimental tree that asks whether a Welvet net can behave like a **synthetic organism** — an embodied loop with its own clock — instead of a compressed calculator that waits for a prompt.

| Piece | What it is | Where |
|-------|------------|--------|
| **Welvet** | Engine: layers, dtypes, quants, SIMD/WebGPU, TrainModes | `github.com/openfluke/welvet` |
| **Lucy** | Measuring math only (Score, SoftAcc, LPD). No datasets. | `welvet/lucy` |
| **Tide** | Serve+train runtime: permute matrix, dash, ocean, checkpoints | `chaosglue/tide` |
| **AAI** (this tree) | Lucy races, cameral sandwiches, ARC probes, credit-mode hunts | `welvet/apps/aai` → [`openfluke/aai`](https://github.com/openfluke/aai) |

Tide hosts that are **not** in this folder (`live_mnist`, `live_gpt`, `quick_sprint`) still speak Lucy. They are the same question on a real dataset. AAI is where the protocols were invented and where train-mode families get tortured.

```
Score = Throughput × Availability × Acc / 10,000
```

**Acc** is hard argmax. **SoftAcc** is serve-confidence. Do not mix those sentences. Sparse can win Score (skip-GEMV Avail) and lose Acc. Rivaling backprop means **hard Acc vs StepBP**, not Lucy Score.

---

## What we are looking for

Not a better chatbot. Not a paper that only impresses a faculty board. Not INT8 inference that streams tokens faster while the weights stay dead.

The industry treats edge AI as **inference**: shrink, quantize, phone home to train. Continual-learning papers often cheat with device–cloud loops. That is a fast, private, functionally dead calculator.

We are looking for an architecture that **owns its own perception of time**. True intelligence does not wait for a carriage return. The target is an autonomous, embodied loop that:

1. **Serves** a continuous sensory stream (answers while awake).
2. **Trains** on that same stream without a separate offline job.
3. **Adapts** when the world remaps (label flip, frequency switch, chase→avoid).
4. **Condenses** into a small dtype/quant box without falling into a chance-Acc trap.
5. Does this **on-device** — Raspberry Pi class, no server farm.

That is the **synthetic organism** metric. Lucy Score is live-fit: can the net **learn while it still talks**. Lucy density (LPD) is whether that live-fit survives memory compression. Gold is the trifecta (Acc + Thru + Avail) in ≤20% of the Acc-champ’s RAM.

Proxy gradients (FastProxy, Sparse, Split, …) exist because global backprop is a bad **duty-cycle** default on a live loop. The Pi grids and `MeshTweenSplitSparse` were thermodynamic probes. The **receipt** is [`adaptv2`](adaptv2): Dense chase→avoid→chase, 22 stack-local modes.

---

## The proof — adaptv2 (Dense live-loop)

This is why the lab exists. Same 6-layer Dense (`8→32→64→64→64→32→4`), 15s, workers=1, serve then train every sample, shock at 5s and 10s (`label+2 mod 4`). Full tables: [`adaptv2/README.md`](adaptv2/README.md).

**CPU (`go run .`, SIMD off)** — organism board. Tide is SIMD-only; this host defaults CPU so Avail is honest against the old loom menu.

| | Acc | Avail | Thru | Score | Shock |
|--|----:|------:|-----:|------:|-------|
| NormalBP | 89.2 | 63.7% | 1762 | 1001 | 90→87, barely dips |
| StepBP | 89.4 | **9.8%** | 2181 | 192 | same Acc, serve dead |
| **FastProxy** | **91.2** | **82.7%** | 3492 | 2634 | 94→81, Acc champ |
| **Sparse** | 85.9 | **95.5%** | **7613** | **6241** | 87→82 / 87→87, LPD / Q / gold-std |
| Tween | 46.9 | 20% | 1632 | 154 | 69→**9%** on avoid — control |
| LinearCache | 43.8 | 82.5% | 3484 | 1259 | high Score, dead Acc — trap |

BP **learns**. StepBP **does not run+train** in the Lucy sense (Avail 10%). FastProxy **beats BP Acc** and stays ~83% available. Sparse is not a better Jacobian — it is the synthetic organism (skip-GEMV Avail). Tween / LinearCache / HeadProxyAsync stay graveyard-shaped.

**SIMD (`go run . -simd on`)** — all 22 modes run; Thru ~2× (Sparse ~3×, 17k). Acc ticks up (StepTweenChain **92.5**). Availability **collapses** (NormalBP 64%→**6%**, FastProxy 83%→**8%**) because cheap MatVec lets train own the thread. Sparse still wins LPD (Avail **31%**, Score 4611). Do not quote the SIMD Acc champ as live-fit.

**Do not paste this ranking onto MHA.** `live_gpt` is Embedding → residual causal MHA → vocab CE. FastProxy’s `W_headᵀ` is the LM Dense, not credit through attention. Same mode names, different store — Shakespeare Acc stays BP-family; this Dense table is the proof of *need*, not a universal winner.

---

## How a live loop works

Every honest Lucy / tide job uses the **duty clock**:

```
for each sample (or pulse window):
    ŷ = serve(x)          # InferMs  — the organism is talking
    train(x, target)      # TrainMs  — the organism is adapting
```

SGD that **blocks** serve tanks Availability. A mode that skips most `dW`s (Sparse) raises Avail and therefore Score even if Acc is worse. Fill ticks on the Step\* 1D pipe do not update — that is a schedule, not extra leftover forwards.

**Mid-stream shock** is how we know it is alive:

| Host | Shock |
|------|--------|
| sine benches | frequency `1 → 2 → 3 → 4` |
| `adaptv2` | 5s chase → 5s avoid (`label+2 mod 4`) → 5s chase |
| tide MNIST / Tiny Shakespeare | phase **A → B → A2** (`label = (label+k) mod C`) |

If Acc is high but AdaptPct / KeepLearn die after the flip, the net memorized phase A. If Cons ~75% and Stab ~98% at 13% Acc, the job **collapsed** (wrong and stable). That is not a new family.

**Serve stays a full forward.** Train mode only changes the update (and, for Step\*, the 1D pipe schedule). Mesh\* still needs a Grid; origin-only cubes are hop topology, not 8/27 copies of the sandwich.

---

## Tide — the serve+train framework

Tide is **dataset-agnostic**. A host supplies a `runner.Dataset` and a `chain.Spec` (or `Config.Build`). Dashboard, Lucy metrics, permute matrix, and checkpoints stay the same.

Aligned with the old loom `test41_w_sine_ada_perm` protocol. Measuring math lives in **`welvet/lucy`**. Tide dash / PDF / ocean only **display** it. A new host must not copy the formulas.

### Packages

| Package | Role |
|---------|------|
| `metrics` | Re-export of `welvet/lucy` (`Finalize`, `BuildLPD`, SoftAcc) |
| `permute` | dtype × format × train mode × arch @ **SIMD** |
| `pulse` | live run state for the dashboard |
| `dash` | HTTP + HTML (1s poll). JSON: `/api/live`, `/api/board`, `/api/meta`, `/api/winners`, `/api/start`, `/api/report.pdf` (CORS `*`) |
| `ocean` | tide-of-tides: poll many dashboards, vote best mode/dtype/arch, stitch a master PDF |
| `runner` | concurrent serve + train pulses; mid-stream flips; checkpoints |
| `chain` | CNN / Bi / Tri (or host `Build`) Welvet models |
| `checkpoint` | scores, bests, inflight weights + train offset |
| `report` | pretty cell IDs, heat, Lucy PDF — wraps `lucy.BuildLPD` |

### Matrix (what a tide cell is)

| Dimension | Values |
|-----------|--------|
| Backend | **SIMD only** (CPU-tiled twins were removed) |
| DType | `core.AllDTypes` |
| Quant | `quant.AllFormats` (some hosts pin FormatNone) |
| Modes | Lucy 6 (`sgd`…`step_tween_chain`) **plus** every named Welvet TrainMode. Old Lucy tokens stay frozen so checkpoints resume. |
| Arch | `Config.Cams` (Parallel branches) or `single` / `bicameral` / `tricameral` |

One cell = one dtype × one quant × one train mode × one arch. Finish the matrix → epoch N+1. **DoneIDs skip finished cells**, so adding new modes does not replay epoch-1.

Default MNIST host: one pass over the train split per cell. `CellMin > 0` loops until wall time (timed Lucy races).

### Tide hosts (outside this folder)

| Host | Task | Shock |
|------|------|--------|
| [`live_mnist`](../../../live_mnist) | MNIST 80/20, CNN stem | `label+5 mod 10` |
| [`live_gpt`](../../../live_gpt) | Tiny Shakespeare next-char, causal MHA, **CE** | Caesar on next char (`+5 mod vocab`) |
| [`quick_sprint`](../../../quick_sprint) | synthetic 4-class, **every Welvet layer** as its own tide + ocean | short sprint |

Classification hosts must call **`TrainStackCE`** (softmax − one-hot). MSE on a one-hot stays uniform and never leaves chance. Regression / sine keeps MSE.

```bash
cd live_mnist && go run . -addr :8080 -mode smoke
# open http://127.0.0.1:8080  → Start

cd quick_sprint && go run . -ocean-only -peers http://127.0.0.1:8080
# ocean at :8090 consolidates winners across tides
```

### Ocean

Ocean **does not train**. It polls `/api/board` on many tides and consolidates:

- per-tide Score winner (mode / dtype / arch)
- plurality votes across layers
- Lucy axis champs (hard Acc, Thru, Avail, Score, SoftAcc, …)
- one LPD board across everything linked
- master PDF (`results/ocean-report.pdf`)

That is how you pick a **default recipe** after a layer sprint without reading 20 dashboards by hand.

---

## Lucy metrics — the full board

Source of truth: [`welvet/lucy`](../../lucy). Tide `metrics` is a thin re-export.

### The three pillars (consciousness)

| Pillar | Symbol | Formula | Meaning |
|--------|--------|---------|---------|
| **Hard Acc** | Acc | argmax correct / outputs × 100 (`avg_accuracy`) | Did it **learn**. Rival vs StepBP. Acc champ is the RAM reference. |
| **Throughput** | T | outputs / wall seconds | Actions per second while the sweep is live. |
| **Availability** | Avail | `InferMs / (InferMs + TrainMs) × 100` | Duty cycle: can you still talk to it while it trains. |

**Consciousness Q** = geometric mean of RelAcc, RelThru, RelAvail vs **learner** peaks. A learner is Acc keep ≥70% of the Acc champ. Chance-Acc tiny dtypes do **not** set the Thru/Avail peaks (that would let a trap define “fast”).

Consciousness radar = `row.Consciousness()` → Acc/Thru/Avail keep in `[0,1]`.

### Live-fit

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Lucy Score** | `T × Avail × Acc / 10,000` | Live-fit. SoftAcc is **not** this term. SGD that blocks serve dies here. |
| **ZeroDowntime** | `Acc × Avail / 100` | Accurate and still serving (no T). |
| **Realtime** | `T × Avail / 100` | Fast duty cycle (no Acc). |
| **AccThru** | `Acc × T / 100` | Accurate and fast (no Avail). |

Older sine comments that say `Score = T × Avail × SoftAcc` are **stale**. Current engine: **hard Acc**.

### Serve-confidence (not Score)

| Metric | Formula | Meaning |
|--------|---------|---------|
| **SoftAcc** | `100 × (1 − \|pred−target\| / scale)` clamped `[0,100]` | How sure the serve is. |
| SoftAcc sine | scale **0.10** (`SoftAccOne`) | Continuous targets. |
| SoftAcc class | scale **1.0** on p(true) (`SoftAccProb`) | ≈ `100 × p(true class)`. Chance MNIST ~10%; Tiny Shakespeare ~1.5%. |
| **AdaptPct** | mean SoftAcc in the first N pulse windows after each phase switch | Recovery after the shock. |
| **KeepLearn** | late SoftAcc still rising vs peak (tide dash) | Not a plateau after a lucky first second. |
| **Consistency** | % of windows with SoftAcc ≥ 10 | High + low Acc = collapsed job. |
| **Stability** | low SoftAcc variance after switches | High + dead Acc = stuck. |

### Learning speed

| Metric | Meaning |
|--------|---------|
| `time_to_acc25_sec` / `time_to_acc50_sec` | Wall seconds until a 1s **hard Acc** window hit ≥25% / ≥50% (lower better) |
| `acc_per_sec` | Final SoftAcc ÷ duration |
| `mobile_acc_per_sec` | Acc/sec ÷ model MiB |

### Memory / density (synthetic organism in a small box)

| Metric | Formula | Meaning |
|--------|---------|---------|
| **LPD** | `Q × shrink` vs Acc-champ RAM; **0** unless RelAcc ≥ 70% | Memory intelligence. Weeds Score/MiB traps. |
| shrink | `min(AccChampRAM / thisRAM, 32)` | How much smaller than the Acc champ. |
| **Gold** | all 3 pillars ≥80% **and** RAM ≤20% of Acc champ | Trifecta in a small box. |
| **Gold-std** | Acc ≥80% plus Thru or Avail ≥80%, then smallest then fastest | Two-or-more of the trifecta. |
| **Near** | keep ≥50% RAM of champ, strong pillars | Close, not gold. |
| **Trap** | RAM ≤20% of Acc champ **and** Acc keep <70% | Binary / chance Acc looking “dense.” |
| **MobileScore** | `Score / WeightMiB` | **Trap.** Do not use for goldilocks. INT8 at chance Acc wins this. |

Memory density radar = `row.MemoryDensity()` → pillars × shrink (traps sit at origin).

A **Pareto front** is the undominated edge of Acc ↔ RAM ↔ Availability. Dominated cells fall off. There is rarely one winner column — there is a goldilocks edge.

### Lucy axis champs (tide dash / ocean)

Every finished tide board names a champion on:

`hard_acc` · `throughput` · `availability` · `score` · `soft_acc` · `acc_thru` · `realtime` · `adapt` · `keep_learn` · `acc_per_sec` · `time_to_50` · `consistency` · `stability` · `mobile_score`

Read **all** of them. The Score champ is often Sparse. The Acc champ is often StepBP or FastProxy. The LPD champ is the organism in a box.

### Honesty (do not write these)

- “Sparse beat backprop” — Sparse skipped hidden GEMVs; check Acc.
- “We beat PyTorch” — this board is live-fit, not ImageNet.
- “MobileScore is the edge winner” — that is the binary trap; use LPD.
- “Mesh\* trained 27 copies” — origin-only cubes disable extra cells.
- “Step\* is leftover forwards” — it is a 1D systolic pipe (`IsLineStep` / `TrainLine`).
- Quoting w2a Test49 as a Lucy race — Test49 is permutation **smoke**.

---

## Train modes (what the organism can update with)

**29 named** `parallel.TrainMode`s (`AllNamedTrainModes()`, Inherit omitted). Persistence uses the full `String()`. Tables use `Short()`:

```
[T]=Tween  [S]=Split  [FP]=FastProxy  [L]=Linear  [HP]=HeadProxy
```

| Helper | Count | Who |
|--------|------:|-----|
| `AllNamedTrainModes()` | 29 | Test49 / test50 set |
| `AllStackLocalTrainModes()` | 22 | no Inherit, no Mesh\* — default `adaptv2 -modes all` |
| `AllCreditTrainModes()` | 16 | Split/Alt + Step\* credit twins (no Mesh) |
| `AllMeshCreditTrainModes()` | 4 | Mesh Split / Alt / FastProxy / Sparse |
| `IsLineStep()` | 11 | 1D pipe (StepBP, StepTween\*, six Step\* credit twins) |

**Step\*** = schedule on a Sandwich (one sample enters child 0 per tick; output is the sample that entered *D* ticks ago; fill ticks do not update). Same family Jacobian / proxy as the non-Step twin. **Mesh\*** = volumetric grid walk. No Mesh HeadProxy / Linear / LinearCache / HeadProxyAsync.

| Family | What the gap does |
|--------|-------------------|
| Backprop (StepBP rival) | Real chain rule `Jᵀ` |
| Tween | Broadcast `P(g_y)`, half LR — blind to `Wᵀ` sign; dead on sine |
| TweenChain | Chain rule (= BP on a Sandwich) |
| Split | `g_i = (1/N) P(g_y)` |
| Alt | Split then re-forward then Tween |
| HeadProxy | Head `Jᵀ g_y`; hidden `dW` only |
| FastProxy | `g_proxy = W_headᵀ g_y` (skip act′); DFA with `B := W_headᵀ` |
| Linear | Affine `Wᵀ` walk, skip `⊙ act′` |
| LinearCache | Cached Linear — **dead on sine**. A/B control only |
| HeadProxyAsync | Hidden uses proxy from sample T−1 |
| Sparse | Head + one rotating hidden; **Avail / Score lever**, not a better Jacobian |

Graveyard (do not revive): WScale, TweenHead, TweenTrace, LinearCache as a default. Notes: [`failed.md`](failed.md).

---

## Benches in this tree

Each folder is its own Go module (`replace` → Welvet). JSON results are gitignored.

### Protocol / credit (toys)

| Bench | Question |
|-------|----------|
| [`test41_w_sine_ada`](test41_w_sine_ada) | Original Welvet sine `1→2→3→4` (poly measuring port). Dense / bicameral. |
| [`test41_w_native_cam`](test41_w_native_cam) | Same sine on **native** `Hemispheres` sandwiches (Bi/Tri/Quad) + Mix BranchModes. Home of credit recipes vs StepBP. |
| [`test47`](test47) | Tween vs Split vs Alt on xor / sine / copy × every layer kind. |
| [`test48`](test48) | Credit modes × **every layer** × xor/sine/copy × 34 dtypes. Combinatorial sweep. |
| [`test50`](test50) | Deep **FP32** Lucy **mode race**: all named modes × Dense/Bi/Tri × 1³/2³/3³ origin-only. Rival = Acc vs StepBP. |
| [`adaptv2`](adaptv2) | **The proof.** Loom **[2]** Dense chase→avoid→chase, 22 stack-local modes, LPD. CPU: FastProxy Acc champ, Sparse organism, StepBP Avail 9.8%. SIMD: Thru up, Avail dies except Sparse. |

test48 answers “does FastProxy/Sparse even work on CNN/Mamba/…”. test50 answers “who wins a timed race on the sandwich the modes were designed for.” **adaptv2 is why the other benches exist** — Dense live-fit first, then ask whether the same names survive another layer kind.

### ARC-AGI (harder world)

Few-shot is **not** the protocol. One sandwich trains on **all** training demos, then we ask Fit / TrainSolve / EvalSolve. Grids pad to 30×30 → **902-d**. A cameral is **n parallel hemispheres**, not extra depth.

| Bench | Question |
|-------|----------|
| [`test44`](test44) | ARC-AGI × native camerals, default StepTweenChain. Did pixels move? Did it solve train tests? Eval zero-shot? |
| [`test45`](test45) | test44 plus **hierarchical consolidation**: keep nets whose pixels moved, merge families, retrain the super sandwich. |
| [`test46`](test46) | Same ARC sandwich, **StepTweenSplit** (output gap split 1/N onto every leaf). |

[`ARC-AGI`](ARC-AGI) / [`ARC-AGI2`](ARC-AGI2) are the public task dumps. They are not Lucy hosts.

### What each ARC column is looking for

| Column | Looking for |
|--------|-------------|
| FitPix / TrainPix | Did the net reconstruct demos (not chance coloring). |
| TrainSolve | Held-out test grids of **training** tasks. |
| EvalSolve | Evaluation split, never trained — transfer, not leakage. |

ARC is where Sparse’s Acc risk shows up. Toys can lie; 902-d grids less so.

---

## Cameral sandwiches

A hemisphere is an independent sub-net with its own weights sharing one `x`. **Not** a second hidden size and not sprites.

```
x → Dense stem → ┬ Hemi 1 (own W, own TrainMode)
                 ├ Hemi 2
                 └ Hemi n     merge add|avg|concat|filter
                          → Dense head → ŷ
                          loss → g_y → each branch’s update
```

Stem and head are shared. Credit modes need a **head Jacobian** (or `W_headᵀ` proxy). That is why Lucy jobs are sandwiches even at cameral count 1 (mid is one Dense, not Parallel). **Mix** stamps `SetBranchModes` so left can be StepBP while right is TweenChain — one loss on the merge. Grid `training.Step` is one mode for the whole net; Mix needs `TrainStackMSE` / `TrainStackCE`.

---

## What “winning” means (checklist)

Use this when you read a JSON dump or a tide dash:

1. **Did it learn?** Hard Acc vs StepBP (and vs chance). AccΔ is the rival sentence.
2. **Did it stay awake?** Availability. If train blocks serve, Score is a lie about organisms.
3. **Did it recover after the shock?** AdaptPct / KeepLearn / SoftAcc after the flip.
4. **Is Score a duty-cycle trick?** Sparse / skip-GEMV → rank Acc and SoftAcc too.
5. **Can it shrink?** LPD / Gold / Trap. Ignore MobileScore for goldilocks.
6. **Did the job collapse?** ~13% Acc + high Cons/Stab on sine = dead, not a new mode.
7. **Is the clock the one you think?** Step\* pipe ≠ Mesh\* grid ≠ leftover forwards.

The organism we want is **gold**: Acc, throughput, and availability all ≥80% of learner peaks, at ≤20% of the Acc-champ’s RAM, still adapting after the world remaps — on a box that never phones home.

---

## Quick start

```bash
cd welvet/apps/aai/adaptv2
go run .              # CPU proof board (default -simd off)
go run . -simd on     # SIMD: faster, Avail tanks except Sparse

cd ../test50
go run . -grids 1 -duration 1s   # FP32 mode race, 1³ only

cd ../test41_w_native_cam
go run . -modes stepbp,fastproxy,sparse -duration 10s
```

Tide (MNIST smoke):

```bash
cd ../../../live_mnist   # chaosglue/live_mnist
go run . -addr :8080 -mode smoke -autostart
```

Feature-book chapters: [§66 Lucy](https://openfluke.github.io/welvet/chapters/66-lucy.html) · [§67 TrainMode](https://openfluke.github.io/welvet/chapters/67-train-modes.html) · [§68 Cameral](https://openfluke.github.io/welvet/chapters/68-cameral.html) · [§69 density](https://openfluke.github.io/welvet/chapters/69-lucy-density.html).
