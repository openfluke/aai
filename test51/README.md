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

So you no longer wait for every `lr=0.02` across all modes before seeing higher LRs on the architecture you care about.

## Quick start

```bash
cd apps/aai/test51
go test .
go run .                    # dense + float32 × all modes × funny LR × cams × grids × challenges
# open http://127.0.0.1:5151 → Start
```

Defaults keep the leaf board small per tree (typically `8 LR × 3 cams × 3 grids = 72` rows). Full matrix:

```bash
go run . -permute            # layers=all dtypes=all …
# or shrink further:
go run . -modes NormalBP,TweenSplitSparse -challenges chase -lrs 0.02,2 -cams 1-3 -grids 1-3
```

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-modes` | `all` | all named train modes, or csv |
| `-lrs` | `funny` | `0.02…1e6` ascending, or csv |
| `-layers` | `dense` | `dense\|…\|all` |
| `-dtypes` | `float32` | `float32\|all\|csv` |
| `-challenges` | `all` | chase/flee/collect/teleport/shock |
| `-cams` | `1-3` | single → tricameral |
| `-grids` | `1-3` | `1×1×1` → `3×3×3` |
| `-permute` | false | layers=all dtypes=all + funny LR + all challenges + cams/grids |
| `-duration` / `-after-freeze` / `-after-train` | `3s` / `2s` / `3s` | phase walls |
| `-ckpt` | `test51_ckpt` | progress + history + results |
| `-addr` | `:5151` | dash (`""` disables) |

## Dashboard

- **Active tree** banner — mode / layer / dtype / challenge + tree & leaf progress
- **Board** — only current tree leaves (LR × cam × grid) with Acc / Score / Δacc / Δ%
- **Consolidation reports** — finished trees (best leaf); board clears after each
- **Playfield** — simple 2D canvas (agent / goal / think orbs), not a noisy Three.js stage

## Self-improve phases (per leaf)

1. **train** → 2. **after_freeze** (no updates) → 3. **after_train** (more SGD)  
   Δacc / Δ% = after_train vs train.
