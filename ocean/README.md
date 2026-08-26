# aai ocean — Tide-of-tides for the farm

Polls remote Tide dashboards and consolidates Lucy winners. **No training.**
Peer IPs live in **`.env`** (gitignored).

## Quick start

```bash
cd apps/aai/ocean
cp .env.example .env          # edit IPs / ports
go run .
# → http://localhost:8090
```

Needs sibling `chaosglue/tide` (and welvet/webgpu via replace) next to `welvet/`.

## `.env`

| Key | Meaning |
|-----|---------|
| `OCEAN_ADDR` | listen (default `0.0.0.0:8090`) |
| `OCEAN_TITLE` | UI title |
| `TIDE_PEERS` | comma list of tide origins |

Named peers (nice labels on the board):

```bash
TIDE_PEERS=m4=http://192.168.0.22:8080,m5_hi=http://192.168.0.244:8082
```

Or bare URLs (`tide-1`, `tide-2`, …):

```bash
TIDE_PEERS=http://192.168.0.22:8080,http://192.168.0.244:8082
```

Each peer must be a running Tide with `/api/board` (test53 lo `:8080`, hi `:8082`, neg `:8081`, etc.).

Registration is **off by default** (`StaticOnly`) so old Pi/quick_sprint workers cannot append themselves via `POST /api/register`. Set `OCEAN_ALLOW_REGISTER=true` only if you want that.

`.env` `TIDE_PEERS` wins over a leftover shell `TIDE_PEERS`.

## Flags

Same knobs override env: `-addr`, `-title`, `-peers`, `-out`, `-allow-register`.
