# test53 — dayroute × layer × mode × dtype × funny-LR (MT + Tide + LPD)

Synthetic **human daily life** on an XY grid — not xor / sine / copy / remap.

## Task: `dayroute`

8×8 apartment. Schedule each day:

`wake → bath → breakfast → work → lunch → gym → couch → sleep`

Repeats **5 days**. Each morning places drift ±1 so the route **moves**.
Agent picks **1 of 6 actions**: N / S / E / W / ACT / WAIT.

```
lr↑ → mode → dtype → kind
(~4 × 29 × 34 × 16 ≈ 63k jobs per half; lo or hi)
```
Default `all` layers drop softmax, kmeans, metacognition, sequential (still
selectable by CSV). Embedding stays skipped.

Funny LRs are split across two farms:

| Script | LRs | Tide | Ckpt |
|--------|-----|------|------|
| `./run-docker-lo.sh` | `0.02, 2, 200, 2000` | `:8080` | `test53_ckpt/` |
| `./run-docker-hi.sh` | `20000, 1m, 10m, 100m` | `:8082` | `test53_ckpt_hi/` |
| `./run-docker-neg.sh` | −ramp | `:8081` | `test53_ckpt_neg/` |

Presets via `TEST53_LRS` / `-lrs`:

| Value | Meaning |
|-------|---------|
| `funny-lo` / `lo` | mild half (default) |
| `funny-hi` / `hi` | extreme half |
| `funny` / `all` | full +ramp (lo+hi) |
| `funny-neg` / `neg` | −ramp only |
| `funny±` / `pm` / `signed` | −ramp then +ramp |
| CSV | any list, e.g. `-1m,-0.02,0.02,2` |

## Defaults (`go run .`)

| Knob | Default |
|------|---------|
| layers | **all** (16: dense…rmsnorm; no softmax/kmeans/metacognition/sequential) |
| modes | **all 29** named train modes |
| dtypes | **all 34** |
| lrs | **funny-lo** (0.02…2k; use `./run-docker-hi.sh` for extremes) |
| workers | NumCPU (or 4 via `.env` / Docker) |
| duration | 2s/job |
| Tide | `:8080` |
| resume | true → `test53_ckpt/` on **host** (bind-mounted; survives rebuild) |

Checkpoint files: `progress.json`, `results.json`, `history.json`, `lpd.json`

```bash
# default host path (set in run-docker.sh):
apps/aai/test53/test53_ckpt/

# override:
export TEST53_CKPT_HOST=$HOME/welvet-data/test53_ckpt
./run-docker.sh --build   # safe — rebuilds image only, ckpt stays on host
```

**Never** run `docker compose down -v` — that drops named volumes (test53 uses a host bind, but still avoid `-v`).

## Native

```bash
cd apps/aai/test53
go mod tidy
go run .
```

## Docker (ckpt on HOST)

Needs sibling `tide/` + `webgpu/` next to `welvet/`.

```bash
cd apps/aai/test53
./run-docker-lo.sh --build    # mild LRs → :8080
./run-docker-hi.sh --build    # extreme LRs → :8082
./run-docker-lo.sh --logs
./stop-lo.sh                  # or ./run-docker-lo.sh --stop
```

`--build` **compiles first** (sources bind-mounted into a golang container — not uploaded as context), then Docker only receives `.bin/test53` (tens of MB, not GBs).

Data bind-mount: **`./test53_ckpt/`** (lo) / **`./test53_ckpt_hi/`** (hi) on the host.

Tide: lo `http://localhost:8080` · hi `http://localhost:8082`