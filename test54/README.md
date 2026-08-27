# test54 — deep dayroute × layer × mode × dtype × funny-LR

Same dayroute task as test53, but:

- **Deep stack:** `Dense(stem) → mid×DEPTH → Dense(head)` (default **DEPTH=4** → 6 stages)
- **Lo LRs:** **0.5, 5, 50, 500, 5000**
- **Hi LRs:** **500k, 5m, 50m, 100m**
- **Longer jobs:** default **15s** wall per cell
- **4 modes** — NormalBP + all Sparse (`[T][S]Sparse` / `Step*` / `Mesh*`)
- **Default layer:** `mamba`

## Run one layer (cam1 and/or cam3)

```bash
# m4 — lo band, both cams
./run-docker-lo.sh mamba --build           # cam1 :9080 + cam3 :9100

# m5 — hi band, both cams
./run-docker-hi.sh mamba --build           # cam1-hi :9082 + cam3-hi :9102

./run-docker-lo.sh cam3 mamba --build      # single cam override
./stop-lo.sh                               # stops both cams (default)
./stop-hi.sh
./status-all.sh
```

| Script | LRs | Tide (default both) | Ckpt |
|--------|-----|---------------------|------|
| `./run-docker-lo.sh [layer]` | 0.5 … 5k | `:9080` + `:9100` | `test54_ckpt[_cam3]/<layer>/` |
| `./run-docker-hi.sh [layer]` | 500k … 100m | `:9082` + `:9102` | `test54_ckpt_hi[_cam3]/<layer>/` |

Each cam×band is its **own** compose project. Default starts **cam1 then cam3**.

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
| `TEST54_LRS` / `-lrs` | `funny-lo` or `funny-hi` via scripts |
| `TEST54_DTYPES` | `all` (34) |
| `TEST54_MODES` | 4 modes: sgd + 3× Sparse |
| `TEST54_CAMS` | `both` (cam1 + cam3) |

Jobs ≈ **lo: 5 LR × 4 modes × 34 dtypes ≈ 680** / **hi: 4 LR × 4 × 34 ≈ 544** per layer **per cam** (×2 cams when both).

## Native smoke

```bash
cd apps/aai/test54
go run . -layers mamba -lrs funny-lo -depth 4 -duration 15s -dtypes float32 -modes sgd,TweenSplitSparse -workers 2
```
