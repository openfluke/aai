# test51 — LPD think race · **one architecture tree at a time**

Welvet policy nets **think into themselves**, then act on mock challenges.
Jobs are grouped into **trees** so you can watch one train-mode architecture finish its LR/cam/grid variants before the next mode starts.

While training, each layer process also hosts a **Tide** Lucy dash (second port):

| Layer ports | Surface |
|-------------|---------|
| **:5151+** | test51 tree consolidations (`/report/{n}.pdf`) |
| **:8080+** | Tide Lucy board (`/api/report.pdf`) |

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
| **Tide** | Lucy density / PDF reports fed live from the same leaves |

## Config via `.env`

```bash
cp .env.example .env
# edit TEST51_MODES=NormalBP,FastProxy
# edit TEST51_LAYERS=dense,dense-wide   # which layers ./start.sh launches
```

| Env | Default | Meaning |
|-----|---------|---------|
| `TEST51_MODES` | `all` | csv train modes (`NormalBP`, `sgd`, `FastProxy`, …) |
| `TEST51_LAYERS` | `dense` | layers for `./start.sh` (`dense,dense-wide` / `all`) |
| `TEST51_FULL` | `true` | **full permute** of dtypes×challenges×funny-LRs×cams×grids (keeps modes+layers) |
| `TEST51_DTYPES` / `CHALLENGES` / `LRS` / `CAMS` / `GRIDS` | `all` / `funny` / … | only used when `TEST51_FULL=false` |
| `TEST51_PORT_BASE` | `5151` | dash port base (`+0,+1,…` per layer) |
| `TIDE_PORT_BASE` | `8080` | Tide port base (`off` disables Tide) |
| `TEST51_CKPT_ROOT` | `test51_ckpt` | each layer → `<root>/<layer>/` |
| `TEST51_AUTOSTART` | `false` | skip Start button |
| `TEST51_RESUME` | `true` | skip done job IDs in that layer’s ckpt |

CLI extras after `--` still reach the binary (`./start.sh dense -- -autostart`).

## 3-server farm (dense · split modes · full permute · autostart)

29 train modes are split across **3 configs**. Each server runs **dense** with
`TEST51_FULL=true` (all dtypes × challenges × funny LRs × cams × grids) and
**autostart** (no Start click).

| Config | Modes (count) | Checkpoint |
|--------|---------------|------------|
| **1** | `sgd`…`TweenSplit` (10) — Lucy core + mesh basics | `test51_ckpt/config1/` |
| **2** | `StepTweenSplit`…`MeshTweenSplit` (10) — Split/Alt family | `test51_ckpt/config2/` |
| **3** | `MeshTweenAlt`…`StepTweenSplitSparse` (9) — mesh/step splits | `test51_ckpt/config3/` |

On each box (chaosglue tree with sibling `welvet/` + `tide/` + `webgpu/`):

```bash
cd welvet/apps/aai/test51

# server A
./run-config.sh 1

# server B
./run-config.sh 2

# server C
./run-config.sh 3
```

That copies `configs/N.env` → `.env` and runs `docker compose up --build -d`
(project `test51-cN`). Training starts immediately.

```bash
./run-config.sh 1 --logs
./run-config.sh 1 --stop
./run-config.sh 1 --local    # no Docker: binary + nohup
```

Dashboards on every server: `:5151` (test51) · `:8080` (Tide).

### Fedora firewall

```bash
./unlock-ports.sh            # open :5151 + :8080 (sudo / firewalld)
./unlock-ports.sh --layers   # also multi-layer :5152-5154 / :8081-8083
./unlock-ports.sh --status
./unlock-ports.sh --lock     # close again
```

## Quick start (no Docker / multi-layer)

```bash
cd apps/aai/test51
cp .env.example .env
./build.sh

./start.sh -i                 # interactive layer picker
./start.sh dense              # one layer
./start.sh dense,dense-wide   # several
./start.sh all                # all four recipes

./status.sh
./stop.sh                     # all
./stop.sh dense               # one
tail -f run/dense.log
```

Each layer is its **own process** with its **own checkpoint folder** and ports:

| Layer | Dash | Tide | Checkpoint |
|-------|------|------|------------|
| dense | :5151 | :8080 | `test51_ckpt/dense/` |
| dense-wide | :5152 | :8081 | `test51_ckpt/dense-wide/` |
| dense-deep | :5153 | :8082 | `test51_ckpt/dense-deep/` |
| dense-deep-wide | :5154 | :8083 | `test51_ckpt/dense-deep-wide/` |

Foreground (single layer, no scripts):

```bash
go run . -layers dense -ckpt test51_ckpt/dense
go run . -modes NormalBP -tide-addr ""
```

## Docker Compose

Prefer `./run-config.sh 1|2|3` (above). Manual:

```bash
cp configs/1.env .env
TEST51_CONFIG=1 docker compose -p test51-c1 up --build -d
```

Build context = chaosglue parent (`welvet` + `tide` + `webgpu`). Maps **5151** + **8080**; ckpt volume `./test51_ckpt/configN`.

## Flags

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `-modes` | `TEST51_MODES` | `all` | named train modes, or csv |
| `-tide-addr` | `TIDE_ADDR` | `0.0.0.0:8080` | Tide dash (`""` off) |
| `-addr` | `TEST51_ADDR` | `0.0.0.0:5151` | test51 dash |
| `-lrs` | `TEST51_LRS` | `funny` | `0.02…1e6` or csv |
| `-layers` / `-dtypes` / `-challenges` | matching `TEST51_*` | `all` | matrix axes |
| `-cams` / `-grids` | `TEST51_CAMS` / `GRIDS` | `1-3` | cameral × mesh |
| `-duration` / `-after-freeze` / `-after-train` | `TEST51_*` | `3s` / `2s` / `3s` | phase walls |
| `-ckpt` | `TEST51_CKPT` | `test51_ckpt` | progress + history + results |
| `-autostart` | `TEST51_AUTOSTART` | `false` | skip Start gate |

## Dashboard

- **Active tree** banner — mode / layer / dtype / challenge + tree & leaf progress
- **Board** — only current tree leaves (LR × cam × grid) with Acc / Score / Δacc / Δ%
- **Consolidation reports** — per-challenge; click → full report + PDF
- **LPD sub-leaderboards** — one board per challenge
- **Tide (:8080)** — Lucy leaderboard + `/api/report` / `/api/report.pdf`

## Self-improve phases (per leaf)

1. **train** → 2. **after_freeze** (no updates) → 3. **after_train** (more SGD)  
   Δacc / Δ% = after_train vs train.
