# Test 46 — whole-net TweenSplit

Same ARC-AGI sandwich / Lucy protocol as **test44**, but training is **Tween**,
not TweenChain.

Old `ModeTween` walks the stack **layer by layer** and only the leaves whose
shape matches the 902-d output gap actually move (stem/hemispheres often skip).
**StepTweenSplit** measures the gap **once at the output**, then splits that
same measurement `1/N` across every trainable leaf (stem, each hemisphere,
head). Each leaf uses its real forward activation. No chain rule, no Jacobian
through the sandwich.

Default sweep: Dense **1-cameral through 8-cameral**.

```bash
cd apps/aai/test46
go run .                          # Dense, 1..8-cameral, all tiles
go run . -only 3                  # just tricameral
go run . -only 3 -layers all      # every kind except Dense, tricameral each
go run . -n 20 -item-time 50ms    # short smoke
go run . -camerals 8 -layers cnn2,mha,residual
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-n` | `0` (all) | cap tasks per split |
| `-camerals` | `8` | max hemispheres (`cam-min`..N plus Dense) |
| `-cam-min` | `1` | first cameral count |
| `-only` | `0` | exactly N hemispheres — no Dense, no sweep |
| `-item-time` | `125ms` | TrainStackMSE budget per demo |
| `-layers` | empty | comma/space list, or `all` (except dense) |
| `-hidden` | `64` | hidden width |
| `-set` | `agi1` | `agi1` \| `agi2` |

Writes `test46_results.json` (gitignored). Mode is `StepTweenSplit`.
