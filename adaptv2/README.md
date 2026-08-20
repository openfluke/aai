# adaptv2 — the Dense live-loop proof

Loom `lucy_bloom_rivers` menu **[2]** (chase → avoid → chase on a 6-layer
Dense) on Welvet Stack + every **stack-local** train mode, scored with
**Lucy density (LPD)** not Score/MiB.

This is the receipt for AAI: **backprop learns this toy; it is not the
serve+train default.** Credit modes are different physics. Exporting the
ranking to MHA / GPT is a different store — see the [AAI README](../README.md).

## Protocol

- Net: **8→32→64→64→64→32→4** Dense (float32)
- **5s Chase** → **5s Avoid** (`label+2 mod 4`) → **5s Chase**
- Serve then train every sample (Lucy duty clock, `RUSAGE_THREAD`)
- Hard Acc every **1s**
- `Score = Throughput × Availability × Acc / 10,000`
- Board: `lucy.BuildLPD`
- Default `-modes all` = `AllStackLocalTrainModes()` (22 modes, **no Mesh\***)
- Default **`-simd off`** (loom CPU board). Tide hosts are SIMD-only; here SIMD is opt-in.

`-modes named` adds Mesh\* on a 1³ origin cell. `-modes step` is the 1D pipe
only. `-modes loom` is the original six: NormalBP, StepBP, Tween, TweenChain,
StepTween, StepTweenChain.

```bash
cd welvet/apps/aai/adaptv2
go run .                 # 15s, CPU, 22 modes  — the live-fit proof
go run . -simd on        # same board, BackendSIMD
go run . -simd both      # twins (Avail is not comparable across workers>1)
```

Tables use `[T]=Tween  [S]=Split  [FP]=FastProxy  [L]=Linear  [HP]=HeadProxy`.
JSON keeps full names.

---

## This run — 22 modes, 15s, workers=1

Same box, two backends. Acc rival = **hard Acc vs NormalBP / StepBP**. Score /
LPD punish duty-cycle death.

### CPU (`go run .`, SIMD off) — the organism board

Acc champ **FastProxy 91.2%**. Score / Q / gold-std **Sparse** (Avail 95.5%,
Thru 7613). NormalBP Acc **89.2** at Avail **63.7%**. StepBP Acc **89.4** at
Avail **9.8%** — same learner, dead serve.

| Mode | Thru | Avail | Acc | Soft | Score | 1st flip | 2nd flip |
|------|-----:|------:|----:|-----:|------:|----------|----------|
| NormalBP | 1762 | 63.7% | 89.2 | 35.4 | 1001 | 90→87 0s | 90→86 0s |
| StepBP | 2181 | **9.8%** | 89.4 | 35.7 | 192 | 90→87 0s | 91→88 0s |
| Tween | 1632 | 20.0% | 46.9 | 26.2 | 154 | 69→51 3s | 57→65 2s |
| TweenChain | 1802 | 63.5% | 89.7 | 35.7 | 1027 | 90→88 0s | 90→88 0s |
| StepTween | 1907 | 9.1% | 49.4 | 26.7 | 86 | 70→61 4s | 61→74 2s |
| StepTweenChain | 2172 | 9.8% | 89.8 | 35.8 | 192 | 90→85 0s | 93→89 0s |
| TweenSplit | 3296 | 82.0% | 86.7 | 35.9 | 2341 | 91→70 0s | 92→62 0s |
| StepTweenSplit | 3429 | 82.6% | 87.5 | 36.1 | 2478 | 91→70 0s | 94→65 0s |
| TweenAlt | 964 | 33.8% | 75.3 | 35.1 | 246 | 80→77 1s | 82→51 0s |
| StepTweenAlt | 1114 | 5.1% | 51.2 | 27.3 | 29 | 70→59 3s | 64→64 2s |
| HeadProxy | 3374 | 81.8% | 87.3 | 34.1 | 2410 | 90→73 0s | 90→80 0s |
| Linear | 3034 | 80.2% | 89.9 | 34.7 | 2188 | 93→75 0s | 94→74 0s |
| **FastProxy** | 3492 | **82.7%** | **91.2** | 34.8 | 2634 | **94→81** 0s | **94→83** 0s |
| LinearCache | 3484 | 82.5% | 43.8 | 27.0 | 1259 | 59→55 4s | 55→54 2s |
| HeadProxyAsync | 3358 | 81.8% | 56.0 | 29.4 | 1537 | 59→56 2s | 62→ — |
| **Sparse** | **7613** | **95.5%** | 85.9 | 35.3 | **6241** | 87→82 0s | 87→87 0s |
| Step HeadProxy | 3447 | 82.5% | 89.4 | 34.8 | 2541 | 92→79 0s | 93→84 0s |
| Step Linear | 3018 | 80.1% | 88.4 | 34.1 | 2136 | 94→61 0s | 94→71 0s |
| Step FastProxy | 3402 | 82.3% | 89.1 | 34.6 | 2495 | 92→77 0s | 92→82 0s |
| Step LinearCache | 3381 | 81.9% | 44.9 | 26.9 | 1242 | 63→ — | 44→55 1s |
| Step HeadProxyAsync | 3231 | 81.5% | 48.8 | 28.1 | 1287 | 67→ — | 44→ — |
| Step Sparse | 7437 | 95.4% | 85.4 | 35.0 | 6058 | 85→85 0s | 86→82 0s |

LPD (CPU): Acc champ FastProxy / off. Score + live-fit Q + gold-std **Sparse**
(LPD 0.98). NormalBP band **acc** (Q 0.53). StepBP Q **0.31**. HeadProxyAsync
LPD **0** (Acc keep 61%). LinearCache is the control (high Score, dead Acc).

Hard Acc by second (CPU) — chase │ avoid │ chase:

| Mode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|------|--:|--:|--:|--:|--:|--:|--:|--:|--:|---:|---:|---:|---:|---:|---:|
| NormalBP | 81 | 88 | 89 | 89 | 90 | 87 | 91 | 90 | 91 | 90 | 86 | 92 | 90 | 91 | 92 |
| StepBP | 82 | 88 | 89 | 90 | 90 | 87 | 91 | 90 | 90 | 91 | 88 | 91 | 91 | 92 | 92 |
| Tween | 44 | 64 | 65 | 68 | 69 | **9** | **8** | 24 | 51 | 57 | 15 | 37 | 65 | 62 | 65 |
| FastProxy | 82 | 93 | 94 | 94 | 94 | **81** | 94 | 94 | 93 | 94 | **83** | 93 | 93 | 93 | 93 |
| Sparse | 81 | 86 | 83 | 84 | 87 | **82** | 84 | 87 | 87 | 87 | 87 | 87 | 89 | 89 | 89 |
| LinearCache | 21 | 40 | 53 | 58 | 59 | 19 | 26 | 31 | 44 | 55 | 30 | 50 | 54 | 59 | 58 |

Tween **collapses on avoid**. FastProxy dips and recovers at the Acc ceiling.
Sparse barely flinches. BP barely dips — and still loses the duty clock.

### SIMD (`go run . -simd on`) — faster GEMV, worse organism

SIMD is linked and **all 22 modes run**. Thru roughly **2×** (Sparse **3.3×**:
7613 → 16967). Acc ticks up a couple points (StepTweenChain **92.5** Acc champ).

Availability **falls through the floor** for chain-rule and even FastProxy
(NormalBP **63.7% → 6.0%**, FastProxy **82.7% → 7.9%**). Train owns the thread
once MatVec is cheap. Sparse is the exception: Avail **31%**, Score **4611**,
still LPD / Q / gold-std.

| Mode | Thru | Avail | Acc | Score |
|------|-----:|------:|----:|------:|
| NormalBP | 3776 | **6.0%** | 91.6 | 208 |
| StepBP | 4475 | 6.8% | 91.9 | 280 |
| TweenChain | 3584 | 5.8% | 91.9 | 191 |
| StepTweenChain | 4415 | 6.8% | **92.5** | 279 |
| FastProxy | 4940 | 7.9% | 91.8 | 357 |
| **Sparse** | **16967** | **31.0%** | 87.6 | **4611** |
| Step Sparse | 16338 | 29.6% | 86.6 | 4186 |
| TweenSplit | 4821 | 9.2% | 88.0 | 388 |
| LinearCache | 4960 | 7.4% | 49.6 | 182 |

Do **not** quote the SIMD Acc champ (StepTweenChain) as the live-fit winner.
Avail 6.8%. Sparse still the organism. Tide defaults to SIMD because it is
sweeping dtypes on a dash, not because SIMD raised Lucy Score here.

---

## How to read it

1. **BP works as a learner** on this Dense toy (≈89–92% Acc, tiny flip dips).
2. **BP is not the run+train default.** StepBP Avail 9.8% CPU / 6.8% SIMD.
   NormalBP is usable on CPU (64% Avail) and dies under SIMD (6%).
3. **FastProxy is the Acc rival** on CPU (91.2 vs BP 89.2) at 83% Avail.
4. **Sparse is the organism** (LPD / Q / gold-std both backends). Acc trails
   FastProxy; shock stability and skip-GEMV Avail are the story. Not “better
   backprop.”
5. **Tween / LinearCache / HeadProxyAsync** stay controls (shock death, stale
   cache, LPD 0).
6. **Step\* vs non-Step** share the family update on Stack; StepBP/StepTween
   still wreck Avail (pipe + chain rule). Step credit twins stay in the Split
   Avail class on CPU.
7. **Layer kind:** this ranking is **Dense GEMV leaves + a 4-way head**. Causal
   MHA (`live_gpt`) is a different Jacobian — don’t paste this table onto
   Shakespeare.

Full pulse dumps: `adaptv2_results.json` (gitignored; last `-simd` run overwrites).
