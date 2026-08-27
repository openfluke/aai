# First-pass cut — weak / non-temporal on dense cam3

These are filtered out of `TEST53_MODES=all` and the default mode list
(`main.go` → `removedTrainModes`).

| Token | Welvet name | Short |
|-------|-------------|-------|
| tween | Tween | [T] |
| tween_chain | TweenChain | [T]Chain |
| step_tween | StepTween | Step[T] |
| TweenAlt | TweenAlt | [T]Alt |
| StepTweenAlt | StepTweenAlt | Step[T]Alt |
| MeshBP | MeshBP | MeshBP |
| MeshTweenAlt | MeshTweenAlt | Mesh[T]Alt |
| MeshTweenChain | MeshTweenChain | Mesh[T]Chain |

Leaves **21** modes (was 29).
