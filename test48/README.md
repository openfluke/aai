# Test 48 — test41 credit modes × every layer × xor / sine / copy

Did FastProxy / Sparse beat StepBP on a 10s **sine** sandwich? This is the
same Lucy board on **every welvet layer kind**, on the toys those sandwiches
are supposed to fit, through **tricameral**.

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

Default `-duration 2s` is the short perm race. For the 10s board:

```bash
go run . -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
```

`-workers 1` is Lucy-honest (Score/Avail). Default workers = NumCPU finishes
the full sweep faster but **shares CPU** so Avail/Score drop.

## Run

```bash
cd apps/aai/test48

# dense only, headline modes, 2s
go run . -layers dense -modes stepbp,tweensplit,headproxy,linear,fastproxy,proxyasync,sparse

# every layer, 1..3 cameral, all 16 modes (big)
go run . 

# xor + sine, CNN + Dense, Bi/Tri
go run . -layers dense,cnn1,residual -tasks xor,sine -cam-min 1 -camerals 3 -duration 2s

# Lucy-honest 10s (slow)
go run . -layers dense -modes stepbp,fastproxy,sparse -duration 10s -switch 2.5s -adapt-windows 10 -workers 1
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
| `-tasks` | `xor,sine,copy` | |
| `-hidden` | `32` | |
| `-lr` | `0.05` | test41 sine used 0.01; toys need 0.05 |
| `-workers` | NumCPU | `1` for comparable Score |
| `-alt-times` | `1` | TweenAlt pairs |

Writes `test48_results.json` (gitignored).

Prints a Score-sorted table per task, then **vs StepBP** deltas for FastProxy /
Sparse / HeadProxy / Linear.

Do not write “we beat backprop” from a 2s XOR lottery. Dense sine at
`-duration 10s -workers 1` is the comparable board.
