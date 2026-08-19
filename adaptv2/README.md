# adaptv2 — dense mid-stream adaptation (Welvet × Lucy density)

Loom `lucy_bloom_rivers` menu **[2]** (chase → avoid → chase on a 6-layer
Dense) rebuilt on Welvet Stack + every stack-local train mode, scored with
**Lucy density (LPD)** not Score/MiB.

Protocol matches the old poly bench:

- Net: **8→32→64→64→64→32→4** Dense
- **5s Chase** → **5s Avoid** (`label+2 mod 4`) → **5s Chase**
- Serve then train every sample (Lucy duty clock)
- Hard Acc buckets every **1s**
- `Score = Throughput × Availability × Acc / 10,000`
- Board folded through `lucy.BuildLPD`

Default `-modes all` is `AllStackLocalTrainModes()` (Step\* credit included,
**Mesh\* omitted** — those need a Grid). `-modes named` adds Mesh\* on a 1³
origin cell. `-modes step` is the 1D pipe only. `-modes loom` is the original
six poly paths: NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain.

## Run

```bash
cd welvet/apps/aai/adaptv2

# full 15s board, CPU, every stack-local mode
go run .

# loom menu [2] six paths, SIMD on/off twins
go run . -modes loom -simd both

# every named mode including Mesh*
go run . -modes named

# pipe only, short probe
go run . -modes step -duration 3s -phase 1s -window 1s
```

Tables print compact train-mode names: `[T]=Tween  [S]=Split  [FP]=FastProxy  [L]=Linear  [HP]=HeadProxy`. JSON keeps the full names.
