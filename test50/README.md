# Test 50 — deep FP32 Lucy race, all named train modes

Who actually wins after a timed Lucy race, not a 2s perm lottery.

Same Lucy harness as test41 / test48: hard Acc, SoftAcc, Avail, AdaptPct,
Tput, Score. Same toys (xor / sine-adapt / copy). Same sandwich
(Dense stem → mid → Dense head). **Float32 only** by default. **Workers=1**
so Score/Avail are Lucy-honest.

Test48 is the combinatorial sweep (layers × dtypes × 2s). This is the
**mode race** — every named update including Mesh*.

Rival = **hard Acc vs StepBP**. Lucy Score rewards throughput × availability
× SoftAcc — Sparse can win Score and lose Acc. Do not mix those sentences.

Not ARC. Every job sits on a **1³ / 2³ / 3³** grid with **one live sandwich at
the origin** (rest disabled — not 8/27 copies). MeshBP = volumetric
`training.Step`, MeshTween = `StepMesh`, MeshTweenChain = `StepTween`; Mesh
Split/Alt/FastProxy/Sparse still credit on the placed stack. Inherit is not
an update.

## Defaults

| Knob | Default | Why |
|------|---------|-----|
| `-dtypes` | `float32` | comparable Lucy board |
| `-layers` | `dense` | the sandwich the modes were designed for |
| `-modes` | `all` | `AllNamedTrainModes()` (**23**, Mesh* included) |
| `-grids` | `1,2,3` | 1³ / 2³ / 3³ origin smoke |
| `-tasks` | `xor,sine,copy` | |
| `-cam-min` / `-camerals` | `1` / `3` | Dense / Bi / Tri |
| `-duration` | **1s** | wall per job (override for a longer race) |
| `-switch` | duration/4 | sine `1→2→3→4` |
| `-adapt-windows` | **10** | pulse after switch |
| `-workers` | **1** | comparable Score |
| `-lr` | `0.05` | same as test48 toys |
| `-hidden` | `32` | |

ETA ≈ jobs × duration / workers. Default is **3 tasks × 3 arches × 3 grids ×
23 modes = 621 jobs × 1s ≈ 10 min**. Pin `-grids 1` or `-only 1` to shrink.

## Run

```bash
cd apps/aai/test50

# the board (float32, Dense, all modes × 1³/2³/3³, 1s/job, workers=1)
go run .

# cubes only, sine, Dense sandwich
go run . -tasks sine -only 1 -grids 1,2,3 -duration 10s

# 1×1×1 only (old wall)
go run . -grids 1

# headline credit vs backprop, 60s
go run . -tasks sine -only 1 -modes stepbp,fastproxy,sparse -duration 60s

# still fp32, but every layer kind (slow)
go run . -layers all
```

Writes `test50_results.json` and `test50_winners.json` (gitignored). End of
run prints an Acc rank and a Score rank per task/arch, plus AccΔ vs StepBP.

## Modes (`-modes all`)

Every named update from `parallel.AllNamedTrainModes()` (**23**). Inherit omitted.

| Token | Mode | Role |
|-------|------|------|
| `stepbp` / `normalbp` | StepBP / NormalBP | **Backprop rival** (same family on Stack) |
| `tween` / `steptween` | Tween | broadcast gap, half LR |
| `tweenchain` / `steptweenchain` | TweenChain | chain rule (= BP on Stack) |
| `meshbp` / `meshtween` / `meshtweenchain` | MeshBP / MeshTween / MeshTweenChain | volumetric scheduler of BP / Tween / TweenChain |
| `tweensplit` / `steptweensplit` | TweenSplit | \(g_i=\frac1N P(g_y)\) |
| `tweenalt` / `steptweenalt` | TweenAlt | Split then Tween |
| `headproxy` | HeadProxy | head \(J^\top g_y\); hidden `dW` only |
| `linear` | Linear | affine \(W^\top\) walk, skip act′ |
| `fastproxy` | FastProxy | \(W_{\mathrm{head}}^\top g_y\), all `dW` only |
| `linearcache` | LinearCache | **dead on sine** — control |
| `proxyasync` | HeadProxyAsync | hidden uses proxy from \(T-1\) |
| `sparse` | Sparse | head + one rotating hidden |
| `meshsplit` / `meshalt` / `meshfastproxy` / `meshsparse` | Mesh Split / Alt / FastProxy / Sparse | Grid-placed credit of those families |

Credit equations: [test48 README](../test48/README.md).

---

## This run — default 1s board (float32, dense, workers=1)

621 jobs: xor / sine / copy × Dense / Bi / Tri × 1³/2³/3³ × 23 modes. Origin-only
grid. **Rival = hard Acc vs StepBP.** Score is Tput × Avail × SoftAcc / 10000.

This is **not** the 10s Lucy race. 1s is enough to rank modes, not enough to
claim a training-law. Do not write “we beat backprop” from xor or from Lucy
Score. Sparse Score is skip-GEMV Avail.

### Copy — Acc vs StepBP is real

Copy is the only toy here where credit **beats StepBP on hard Acc**, across
every arch and every cube. Split / StepTweenSplit / MeshTweenSplit / Alt /
FastProxy sit about **+3 to +12 Acc** over StepBP. StepBP itself is ~68–74%.
Winners rotate (Split vs Alt vs MeshSplit) — the *family* is the story, not
one token.

| Board | Acc winner | Acc | StepBP Acc | AccΔ |
|-------|------------|----:|-----------:|-----:|
| Dense 1³ | TweenSplit | 81.2 | 72.7 | +8.6 |
| Dense 2³ | StepTweenSplit | 82.0 | 72.7 | +9.4 |
| Dense 3³ | MeshTweenSplit | 82.0 | 71.1 | +10.9 |
| Bi 1³ | StepTweenSplit | 80.5 | 73.4 | +7.0 |
| Bi 2³ | MeshTweenAlt | 82.0 | 69.5 | +12.5 |
| Bi 3³ | StepTweenSplit | 82.0 | 72.7 | +9.4 |
| Tri 1³ | StepTweenSplit | 79.7 | 74.2 | +5.5 |
| Tri 2³ | MeshTweenSplit | 78.1 | 70.3 | +7.8 |
| Tri 3³ | MeshTweenAlt | 79.7 | 68.0 | +11.7 |

Score winner on copy is **always** TweenSplitSparse or MeshTweenSplitSparse
(~8k–10k vs StepBP ~300–1600) while Sparse Acc is usually **worse** than
StepBP (−7 to −25). Same Lucy split as test41: Avail, not fit.

Tween / StepTween / MeshTween and LinearCache sit at the bottom (~45–53 Acc).
Controls worked.

### Sine — Acc ceiling, SoftAcc is the knife

Hard Acc is a last-freq eval (\|err\| < 0.15 on the freq-4 pool after 1s of
`1→2→3→4` at 250ms switches). A pack of modes — StepBP, TweenChain,
FastProxy, HeadProxy, Linear, Sparse, MeshBP — often all land at **100%**.
AccΔ vs StepBP is then **+0**. That is not a Sparse win over backprop.

What *does* move:

- **FastProxy SoftAcc ~56–68** vs **StepBP SoftAcc ~42–53** while both are at
  100% hard. That’s the FastProxy vs BP sentence on this board.
- **Even Split / Alt ~48–56 Acc**, Soft ~24. They do not track the switches
  in 1s.
- **Plain Tween / MeshTween ~18–33 Acc**, Soft ~7. Dead on sine, as designed.
- **LinearCache** stays dead (Soft ~5–15).
- Sparse / MeshSparse often **win Score** (and sometimes share the 100% Acc
  cap). Rank them on SoftAcc, not Score.

Dense 3³ is the one sine board where MeshTweenSplitFastProxy is the Acc
headline (100%) and Sparse drops to 87 / 78 Acc — still not “Sparse beat BP”
(StepBP is also 100%).

### Xor — 4-point parking lot

Xor is four bits. Hard Acc is 0/25/50/75/100. Almost the whole pack sits at
**75%** (3 of 4) including StepBP, so AccΔ is +0. SoftAcc still splits
(~84–86 FastProxy/Sparse vs ~80–84 BP vs ~50 Tween). AdaptPct is 0 — no
pulses. Use xor as a smoke, not a ranking.

HeadProxyAsync often **25%** on xor. LinearCache ~50% Acc / terrible Soft.
MeshTween sometimes 0–50.

### Mesh\* vs family

On this origin-only cube, Mesh Split / FastProxy / Sparse usually **match
their stack twin** (same family, extra grid walk). MeshBP tracks StepBP Acc
and sometimes beats it on Score (cheaper volumetric Step). MeshTween /
MeshTweenChain follow Tween, not Split — they are not secret FastProxy.

3³ vs 1³ does **not** explode Acc. Origin-only means the extra cells are
disabled; cube size is a hop topology, not 27 trained sandwiches.

