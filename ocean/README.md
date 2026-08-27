# aai ocean — Tide-of-tides for the farm

Polls remote Tide dashboards **or** scans a local test53 results cache and ranks by **raw Acc**.

## Acc cache mode (test53)

Point at any folder tree of farm copies (each subdir with `results.json`):

```bash
cd apps/aai/ocean
# .env:
#   OCEAN_CACHE=/home/openfluke/Documents/work/aai/test53
#   OCEAN_TITLE=test53 Acc cache
go run .
# → http://localhost:8090/          Acc rank + Acc×Avail + overlapping charts
# → http://localhost:8090/compare   classic CompareReport from disk
# → http://localhost:8090/api/acc
```

Uses top-level **`acc`** only (closed-loop eval). Fit = Acc×Avail/100 (serve+train). Not LPD.

## Live tide mode

```bash
cd apps/aai/ocean
cp .env.example .env          # edit IPs / ports — leave OCEAN_CACHE empty
go run .
# → http://localhost:8090
# → http://localhost:8090/compare
```

Needs sibling `chaosglue/tide` (and welvet/webgpu via replace) next to `welvet/`.

## `.env`

| Key | Meaning |
|-----|---------|
| `OCEAN_ADDR` | listen (default `0.0.0.0:8090`) |
| `OCEAN_TITLE` | UI title |
| **`OCEAN_CACHE`** | recursive scan root for `results.json` (Acc mode; aliases `CACHE_ROOT`) |
| `TIDE_PEERS` | comma list of tide origins (live mode) |
| `TIDE_PEER_HOST` | host/IP for test53 tides (with `TIDE_CAMS`) |
| `TIDE_CAMS` | comma cam counts: `1,3` |
| `TIDE_BANDS` | comma LR bands: `lo,hi` |

If `OCEAN_CACHE` is set, cache Acc mode wins (no live peers required).

## Flags

`-addr`, `-title`, `-peers`, `-out`, `-allow-register`, **`-cache`**.
