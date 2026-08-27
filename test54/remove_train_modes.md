# test54 mode cut — NormalBP + Sparse only

`TEST54_MODES=all` and the default list keep **4** modes:

| Token | Welvet name | Short |
|-------|-------------|-------|
| sgd | NormalBP | sgd |
| TweenSplitSparse | TweenSplitSparse | [T][S]Sparse |
| StepTweenSplitSparse | StepTweenSplitSparse | Step[T][S]Sparse |
| MeshTweenSplitSparse | MeshTweenSplitSparse | Mesh[T][S]Sparse |

StepBP removed. Everything else filtered out (`keptTrainModes` in `main.go`).

## LRs

| Band | Values |
|------|--------|
| lo (`funny-lo`) | 0.5, 5, 50, 500, 5000 |
| hi (`funny-hi`) | 500k, 5m, 50m, 100m |

Jobs ≈ **lo ~680** / **hi ~544** per layer per cam (×4 modes ×34 dtypes).
