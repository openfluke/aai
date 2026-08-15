# Test 47 — Tween vs Split vs Alt (Split↔Tween ping-pong)

Not ARC. XOR / sine / copy, **every layer kind**, six credit modes:

| Mode | Credit |
|------|--------|
| `StepTween` | Broadcast output gap onto every leaf (full gap, half LR). Online name. |
| `Tween` | Same family as StepTween on a Sandwich (same update). |
| `TweenSplit` | Same gap, **split 1/N** across leaves. |
| `StepTweenSplit` | Same family as TweenSplit on a Sandwich. |
| `TweenAlt` | **Split then Tween**, repeat `-alt-times` pairs. Recomputes the MSE gap between phases. |
| `StepTweenAlt` | Same family as TweenAlt on a Sandwich. |

On Stack/Parallel (no Grid), **Step\* and non-Step collapse to the same family update** — the table still runs them as separate jobs so you can see they match. `chain` is actual backprop.

`TweenAlt` with `-alt-times 3` is Split → Tween → Split → Tween → Split → Tween **per sample**. Default is `1` (one Split, then one Tween). Keep it as a third column: on copy it has diverged from Split (residual/rnn/layernorm bicameral Acc lifts on the order of +5–12). XOR Acc is 4-point lottery; sine often sits on Split’s 59.4 plateau. Alt can also NaN where Split is already shaky (mamba/swiglu).

```bash
cd apps/aai/test47
go run .   # default: tween / split / alt
go run . -layers dense,residual,cnn2
go run . -modes tweenalt,steptweenalt -alt-times 4
go run . -modes steptween,tween,tweensplit,steptweensplit,tweenalt,steptweenalt,chain
go run . -budget 3s -camerals 1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-layers` | `all` | including Dense |
| `-camerals` | `2` | 1-cameral .. N |
| `-budget` | `1.5s` | train wall per job |
| `-tasks` | `xor,sine,copy` | |
| `-modes` | `steptween,tween,tweensplit,steptweensplit,tweenalt,steptweenalt` | add `chain` for BP |
| `-alt-times` | `1` | Split→Tween pairs per `TrainStackMSE` (TweenAlt only) |
| `-hidden` | `32` | |
| `-lr` | `0.05` | |

Writes `test47_results.json` (gitignored).
