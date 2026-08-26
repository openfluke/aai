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
| `TIDE_PEERS` | comma list of tide origins |

Compare **two machines** across **all funny-LRs** in each tide archive (test53 puts `|lr=…` on every cell ID). Name peers with a machine prefix:

```bash
TIDE_PEERS=m4-lo=http://192.168.0.22:8080,m5-lo=http://192.168.0.244:8080
TIDE_PEERS=m4-lo=http://192.168.0.22:8080,m4-hi=http://192.168.0.22:8082,m5-lo=http://192.168.0.244:8080,m5-hi=http://192.168.0.244:8082
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
