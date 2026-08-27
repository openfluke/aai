# test54 — deep dayroute × layer × mode × dtype × LR=0.05

Same dayroute task as test53, but:

- **Deep stack:** `Dense(stem) → mid×DEPTH → Dense(head)` (default **DEPTH=4** → 6 stages)
- **Fixed LR:** **0.05** across all modes × dtypes (no funny ramp by default)
- **Longer jobs:** default **15s** wall per cell
- **21 modes** (same cut as test53 `remove_train_modes.md`)
- **Default layer:** `mamba`

## Run one layer

```bash
./run-docker-lo.sh mamba --build          # cam1 → :9080
./run-docker-lo.sh cam3 mamba --build     # tricameral → :9100
./run-docker-hi.sh cam3 mamba --build     # second machine / hi port :9102

./stop-lo.sh cam3
./run-docker-lo.sh cam3 lstm --build
```

| Script | LR | Tide | Ckpt |
|--------|----|------|------|
| `./run-docker-lo.sh [camN] [layer]` | 0.05 | `:9080` / `:9100` | `test54_ckpt[_camN]/<layer>/` |
| `./run-docker-hi.sh [camN] [layer]` | 0.05 | `:9082` / `:9102` | `test54_ckpt_hi[_camN]/<layer>/` |

lo/hi here only split **port + ckpt** so two machines can farm in parallel — same LR.

## Knobs

| Env / flag | Default |
|------------|---------|
| `TEST54_LAYER` / first arg | `mamba` |
| `TEST54_DEPTH` / `-depth` | `4` mid blocks |
| `TEST54_HIDDEN` / `-hidden` | `32` |
| `TEST54_DURATION` / `-duration` | `15s` |
| `TEST54_LRS` / `-lrs` | `0.05` |
| `TEST54_DTYPES` | `all` (34) |
| `TEST54_MODES` | 21 kept modes (`all` skips removed) |
| `TEST54_CAMS` | `1` |

Jobs ≈ **1 LR × 21 modes × 34 dtypes ≈ 714** per layer (cam1). Cam3 same count.

## Native smoke

```bash
cd apps/aai/test54
go run . -layers mamba -lrs 0.05 -depth 4 -duration 15s -dtypes float32 -modes sgd,TweenSplitSparse -workers 2
```
