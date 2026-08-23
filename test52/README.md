# test52 — all train modes on **3×3×3** mesh

Focused smoke: does every `AllNamedTrainModes()` actually **train** on a real
`NewGrid(3,3,3,…)` — not the 1×1×1 mesh path test41 used?

| Mode family | Layout on 3×3×3 |
|-------------|-----------------|
| Most modes (incl. MeshBP, Mesh*Split) | `PlaceStack` at origin |
| **MeshTween / MeshTweenChain** | dense **L-stack** at origin (`L=3`) — `StepMesh` needs this |

Task: XOR (2→1). Per mode: short wall → loss↓ / Acc↑ / \|Δw\| → PASS/FAIL.

```bash
cd apps/aai/test52
go test .
go run .                              # all modes
go run . -modes MeshTween,MeshTweenChain,MeshBP
go run . -duration 3s                 # more wall if a mode is slow
```

Exit **1** if any mode FAIL.
