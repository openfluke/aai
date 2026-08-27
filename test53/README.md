# test53 — dayroute × layer × mode × dtype × funny-LR (MT + Tide + LPD)

Synthetic **human daily life** on an XY grid — not xor / sine / copy / remap.

## Task: `dayroute`

8×8 apartment. Schedule each day:

`wake → bath → breakfast → work → lunch → gym → couch → sleep`

Repeats **5 days**. Each morning places drift ±1 so the route **moves**.
Agent picks **1 of 6 actions**: N / S / E / W / ACT / WAIT.

```
lr↑ → mode → dtype → kind
```

**One layer per run** (~4 × 29 × 34 ≈ **3.9k jobs** per lo/hi half). Finish a layer, stop, start the next:

```bash
./run-docker-lo.sh dense --build
./run-docker-lo.sh convt2 --build   # own ckpt: test53_ckpt/convt2/
./stop-lo.sh
./run-docker-lo.sh mha --build
./list-layers.sh                    # all 16 default layers
```

| Script | LRs | Tide | Ckpt root |
|--------|-----|------|-----------|
| `./run-docker-lo.sh [camN] [layer]` | `0.02, 2, 200, 2000` | `:8080` (cam1) · `:8100` (cam3) | `test53_ckpt[_camN]/<layer>/` |
| `./run-docker-hi.sh [camN] [layer]` | `20000, 1m, 10m, 100m` | `:8082` (cam1) · `:8102` (cam3) | `test53_ckpt_hi[_camN]/<layer>/` |
| `./run-docker-neg.sh [camN] [layer]` | −ramp | `:8081` (cam1) · `:8101` (cam3) | `test53_ckpt_neg[_camN]/<layer>/` |

**Cam × LR band:** run cam1 and cam3 in parallel — separate ports, ckpts, containers:

```bash
./run-docker-lo.sh dense --build          # cam1 (default) → :8080
./run-docker-lo.sh cam3 dense --build     # tricameral mid  → :8100
# ocean compare: cam1-lo=http://host:8080,cam3-lo=http://host:8100
```

Port formula: `8080 + (cam−1)×10 + band` (lo +0, neg +1, hi +2).

Each cam×band is a **separate compose project** (`test53-lo`, `test53-cam3-lo`, …) — they can run in parallel:

```bash
./run-docker-lo.sh dense --build          # cam1 lo → project test53-lo :8080
./run-docker-lo.sh cam3 dense --build     # cam3 lo → project test53-cam3-lo :8100
./status-all.sh                           # see what's up
```

Finished jobs stay in ckpt — restart tide only (no `--build` if binary exists):

```bash
./run-docker-lo.sh dense                  # cam1 lo serves /api/report again
./run-docker-hi.sh dense                  # cam1 hi :8082
```

Default layer is **`dense`** if omitted. Use `TEST53_LAYERS=all` only when you really want all 16 layers in one go (~63k jobs per half).

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
| layers | **`dense`** (one layer; set `all` for full matrix) |
| modes | **all 29** named train modes |
| dtypes | **all 34** |
| lrs | **funny-lo** (0.02…2k; use `./run-docker-hi.sh` for extremes) |
| cams | **1** (single mid; `./run-docker-lo.sh cam3` for tricameral) |
| workers | NumCPU (or 4 via `.env` / Docker) |
| duration | 2s/job |
| Tide | `:8080` |
| resume | true → `test53_ckpt/<layer>/` on **host** (bind-mounted; survives rebuild) |

Native runs auto-ckpt to `test53_ckpt/<layer>/` unless `TEST53_CKPT_FLAT=true` (Docker).

Checkpoint files per layer folder: `progress.json`, `results.json`, `history.json`, `lpd.json`

```bash
# per-layer host paths (set by run-docker-lo.sh):
apps/aai/test53/test53_ckpt/dense/
apps/aai/test53/test53_ckpt/convt2/

# override entire bind mount:
export TEST53_CKPT_HOST=$HOME/welvet-data/test53_ckpt/dense
./run-docker-lo.sh --build
```

**Never** run `docker compose down -v` — that drops named volumes (test53 uses a host bind, but still avoid `-v`).

## Native

```bash
cd apps/aai/test53
go mod tidy
go run . -layers convt2
# ckpt → test53_ckpt/convt2/
```

## Docker (ckpt on HOST)

Needs sibling `tide/` + `webgpu/` next to `welvet/`.

```bash
cd apps/aai/test53
./run-docker-lo.sh dense --build
./run-docker-lo.sh convt2 --build
./run-docker-hi.sh dense --build
./run-docker-lo.sh --logs
./stop-lo.sh                  # or ./run-docker-lo.sh --stop
```

`--build` **compiles first** (sources bind-mounted into a golang container — not uploaded as context), then Docker only receives `.bin/test53` (tens of MB, not GBs).

Tide: lo `http://localhost:8080` · hi `http://localhost:8082`
