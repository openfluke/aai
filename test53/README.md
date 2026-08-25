# test53 — dayroute × layer × mode × dtype (MT + Tide + LPD)

Synthetic **human daily life** on an XY grid — not xor / sine / copy / remap.

## Task: `dayroute`

8×8 apartment. Schedule each day:

`wake → bath → breakfast → work → lunch → gym → couch → sleep`

Repeats **5 days**. Each morning places drift ±1 so the route **moves**.
Agent picks **1 of 6 actions**: N / S / E / W / ACT / WAIT.

Train = teacher-forced oracle routing (walk to place, ACT).  
Eval Acc = closed-loop match on oracle actions; `days_done` = full days completed.

```
kind → mode → dtype     (~20 × 29 × 34 ≈ 19.7k jobs)
```

## Run

```bash
cd apps/aai/test53
go mod tidy
go run .
go run . -layers dense,mha,lstm -dtypes float32 -workers 4 -duration 2s
```

Tide `:8080` · ckpt `test53_ckpt/{results,lpd,progress}.json`
