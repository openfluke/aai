# test54 — deep dayroute × layer × mode × dtype × LR=0.05

Same dayroute task as test53, but:

- **Deep stack:** `Dense(stem) → mid×DEPTH → Dense(head)` (default **DEPTH=4** → 6 stages)
- **Fixed LR:** **0.05** across all modes × dtypes (no funny ramp by default)
- **Longer jobs:** default **15s** wall per cell
- **21 modes** (same cut as test53 `remove_train_modes.md`)
- **Default layer:** `mamba`

## Run one layer (cam1 and/or cam3)

```bash
./run-docker-lo.sh mamba --build           # cam1 only → :9080
./run-docker-lo.sh cam3 mamba --build      # cam3 only → :9100
./run-docker-lo.sh both mamba --build      # cam1 + cam3 (two compose projects)

./run-docker-hi.sh both mamba --build      # second machine: cam1-hi :9082 + cam3-hi :9102

./stop-lo.sh both
./stop-hi.sh cam3
./run-docker-lo.sh both lstm --build
./status-all.sh                            # list cam1/cam3 × lo/hi
```

| Script | LR | Tide | Ckpt |
|--------|----|------|------|
| `./run-docker-lo.sh [cam1\|cam3\|both] [layer]` | 0.05 | `:9080` / `:9100` | `test54_ckpt[_camN]/<layer>/` |
| `./run-docker-hi.sh [cam1\|cam3\|both] [layer]` | 0.05 | `:9082` / `:9102` | `test54_ckpt_hi[_camN]/<layer>/` |

Each cam×band is its **own** compose project (same as test53). `both` starts cam1 then cam3.

lo/hi only split **port + ckpt** so two machines can farm in parallel — same LR.

Ocean peers example (test54 ports):

```
TIDE_PEERS=cam1-lo=http://HOST:9080,cam3-lo=http://HOST:9100,cam1-hi=http://HOST2:9082,cam3-hi=http://HOST2:9102
```

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
