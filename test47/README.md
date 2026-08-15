# Test 47 — Tween vs StepTween vs TweenSplit vs StepTweenSplit

Not ARC. XOR / sine / copy, **every layer kind**, four credit modes:

| Mode | Credit |
|------|--------|
| `StepTween` | Broadcast output gap onto every leaf (full gap, half LR). Online name. |
| `Tween` | Same family as StepTween on a Sandwich (same update). |
| `TweenSplit` | Same gap, **split 1/N** across leaves. |
| `StepTweenSplit` | Same family as TweenSplit on a Sandwich. |

On Stack/Parallel (no Grid), **Step\* and non-Step collapse to the same family update** — the table still runs them as four jobs so you can see they match. `chain` is actual backprop.

```bash
cd apps/aai/test47
go run .   # default: steptween,tween,tweensplit,steptweensplit
go run . -layers dense,residual,cnn2
go run . -modes steptween,tween,tweensplit,steptweensplit,chain
go run . -budget 3s -camerals 1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-layers` | `all` | including Dense |
| `-camerals` | `2` | 1-cameral .. N |
| `-budget` | `1.5s` | train wall per job |
| `-tasks` | `xor,sine,copy` | |
| `-modes` | `steptween,tween,tweensplit,steptweensplit` | add `chain` for BP |
| `-hidden` | `32` | |
| `-lr` | `0.05` | |

Writes `test47_results.json` (gitignored).
