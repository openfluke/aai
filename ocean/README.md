# aai ocean — Tide-of-tides for the farm

Polls remote Tide dashboards and consolidates Lucy winners. **No training.**
Peer IPs live in **`.env`** (gitignored).

## Quick start

```bash
cd apps/aai/ocean
cp .env.example .env          # edit IPs / ports
go run .
# → http://localhost:8090
# → http://localhost:8090/compare   (machine × LR compare + PDF)
```

Needs sibling `chaosglue/tide` (and welvet/webgpu via replace) next to `welvet/`.

## `.env`

| Key | Meaning |
|-----|---------|
| `OCEAN_ADDR` | listen (default `0.0.0.0:8090`) |
| `OCEAN_TITLE` | UI title |
| `TIDE_PEERS` | comma list of tide origins (optional if using auto cam peers) |
| `TIDE_PEER_HOST` | host/IP for test53 tides (with `TIDE_CAMS`) |
| `TIDE_CAMS` | comma cam counts: `1,3` → ports 8080/8100 (lo), 8082/8102 (hi) |
| `TIDE_BANDS` | comma LR bands: `lo,hi` (default `lo,hi`) |

Auto cam peers (easiest for **cam1 vs cam3** compare):

```bash
TIDE_PEER_HOST=192.168.0.22
TIDE_CAMS=1,3
TIDE_BANDS=lo,hi
```

Or explicit URLs:

```bash
TIDE_PEERS=cam1-lo=http://192.168.0.22:8080,cam3-lo=http://192.168.0.22:8100
```

Compare **two machines** (or **cam counts**) across **all funny-LRs** in each tide archive (test53 puts `|lr=…` on every cell ID). Name peers with a machine prefix:

```bash
TIDE_PEERS=cam1-lo=http://192.168.0.22:8080,cam3-lo=http://192.168.0.22:8100
TIDE_PEERS=m4-lo=http://192.168.0.22:8080,m5-lo=http://192.168.0.244:8080
TIDE_PEERS=cam1-lo=...,cam1-hi=http://192.168.0.22:8082,cam3-lo=...,cam3-hi=http://192.168.0.22:8102
```

Or bare URLs (`tide-1`, `tide-2`, …):

```bash
TIDE_PEERS=http://192.168.0.22:8080,http://192.168.0.244:8080
```

Each peer must be a running Tide with `/api/board` and a finished (or live) `/api/report` archive.

| Page | URL |
|------|-----|
| Holistic board | `/` |
| **Compare LR** | `/compare` |
| Compare PDF | `/api/compare.pdf` |
| Ocean PDF | `/api/report.pdf` |

Registration is **off by default** (`StaticOnly`) so old Pi/quick_sprint workers cannot append themselves via `POST /api/register`. Set `OCEAN_ALLOW_REGISTER=true` only if you want that.

`.env` `TIDE_PEERS` wins over a leftover shell `TIDE_PEERS`.

## Flags

Same knobs override env: `-addr`, `-title`, `-peers`, `-out`, `-allow-register`.
