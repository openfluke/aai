# test53 — dayroute × layer × mode × dtype (MT + Tide + LPD)

Synthetic **human daily life** on an XY grid — not xor / sine / copy / remap.

## Task: `dayroute`

8×8 apartment. Schedule each day:

`wake → bath → breakfast → work → lunch → gym → couch → sleep`

Repeats **5 days**. Each morning places drift ±1 so the route **moves**.
Agent picks **1 of 6 actions**: N / S / E / W / ACT / WAIT.

```
kind → mode → dtype     (~20 × 29 × 34 ≈ 19.7k jobs)
```

## Defaults (`go run .`)

| Knob | Default |
|------|---------|
| layers | **all** (dense, mha, lstm, …) |
| modes | **all 29** named train modes |
| dtypes | **all 34** |
| workers | NumCPU (or 4 via `.env` / Docker) |
| duration | 2s/job |
| Tide | `:8080` |
| resume | true → `test53_ckpt/` |

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
./run-docker.sh              # build + up -d, restart: always
./run-docker.sh --logs
./run-docker.sh --status
./run-docker.sh --stop
```

Data is bind-mounted to **`./test53_ckpt/` on the host** — not stuck in the container:

```
test53_ckpt/{progress,results,lpd,history}.json
```

Tide: `http://localhost:8080`
