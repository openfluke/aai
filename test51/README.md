# test51 — LPD think race · **one architecture tree at a time**

Welvet policy nets **think into themselves**, then act on mock challenges.
Jobs are grouped into **trees** so you can watch one train-mode architecture finish its LR/cam/grid variants before the next mode starts.

## Expand order

```
mode → layer → dtype → challenge  →  LR↑ → cams → grid
         └──────── tree identity ────────┘    └── leaves on the board ──┘
```

| Concept | What it is |
|--------|------------|
| **Tree** | Fixed `mode × layer × dtype × challenge` |
| **Leaves** | That tree’s `LR↑ × cams(1–3) × grid(1³–3³)` sweep |
| **Board** | Shows **only the active tree’s leaves** (dozens, not millions) |
| **Report** | When a tree finishes → summary appended, board **clears** |
| **LPD** | **One leaderboard per challenge** — chase never ranks against teleport |

So you no longer wait for every `lr=0.02` across all modes before seeing higher LRs on the architecture you care about.

## Quick start

```bash
cd apps/aai/test51
go test .
./start.sh                  # background on 0.0.0.0:5151 — safe to exit SSH
./stop.sh                   # stop background run
tail -f test51.log          # follow log
go run .                    # foreground (FULL matrix)
# open http://<host-ip>:5151 → Start
```

That is ~**1.4M** jobs (~20k trees × 72 leaves). Resume skips done IDs. To shrink:

```bash
go run . -layers dense -dtypes float32          # ~10k jobs (old “sane” default)
go run . -modes NormalBP -challenges chase -lrs 0.02,2
```

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-modes` | `all` | all named train modes, or csv |
| `-lrs` | `funny` | `0.02…1e6` ascending, or csv |
| `-layers` | `all` | `dense\|…\|all` |
| `-dtypes` | `all` | `float32\|all\|csv` |
| `-challenges` | `all` | chase/flee/collect/teleport/shock |
| `-cams` | `1-3` | single → tricameral |
| `-grids` | `1-3` | `1×1×1` → `3×3×3` |
| `-permute` | false | force full matrix (same as defaults) |
| `-duration` / `-after-freeze` / `-after-train` | `3s` / `2s` / `3s` | phase walls |
| `-ckpt` | `test51_ckpt` | progress + history + results (saved every leaf; `-resume` default skips done IDs) |
| `-addr` | `0.0.0.0:5151` | dash on all interfaces (`""` disables) |

## Dashboard

- **Active tree** banner — mode / layer / dtype / challenge + tree & leaf progress
- **Board** — only current tree leaves (LR × cam × grid) with Acc / Score / Δacc / Δ%
- **Consolidation reports** — click a finished tree → full report tab (graphs + leaf table); **Download PDF** / `/report/{n}.pdf`
- **Playfield** — simple 2D canvas (agent / goal / think orbs), not a noisy Three.js stage

## Self-improve phases (per leaf)

1. **train** → 2. **after_freeze** (no updates) → 3. **after_train** (more SGD)  
   Δacc / Δ% = after_train vs train.
