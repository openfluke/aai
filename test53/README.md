# test53 — dayroute × layer × mode × dtype × funny-LR (MT + Tide + LPD)

Synthetic **human daily life** on an XY grid — not xor / sine / copy / remap.

## Task: `dayroute`

8×8 apartment. Schedule each day:

`wake → bath → breakfast → work → lunch → gym → couch → sleep`

Repeats **5 days**. Each morning places drift ±1 so the route **moves**.
Agent picks **1 of 6 actions**: N / S / E / W / ACT / WAIT.

```
lr↑ → mode → dtype → kind
(~8 × 29 × 34 × 20 ≈ 157k jobs)
```

Sweep order: **all modes × dtypes × layers at LR=0.02**, then LR=2, … up to 100m.

Funny LRs: `0.02, 2, 200, 2000, 20000, 1m, 10m, 100m`

## Defaults (`go run .`)

| Knob | Default |
|------|---------|
| layers | **all** (dense, mha, lstm, …) |
| modes | **all 29** named train modes |
| dtypes | **all 34** |
| lrs | **funny** (8-step ramp) |
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
./run-docker.sh --build   # go build in golang container → tiny image (binary only)
./run-docker.sh           # start existing image
./run-docker.sh --logs
./run-docker.sh --stop
```

`--build` **compiles first** (sources bind-mounted into a golang container — not uploaded as context), then Docker only receives `.bin/test53` (tens of MB, not GBs).

Data bind-mount: **`./test53_ckpt/` on the host**.

Tide: `http://localhost:8080`
