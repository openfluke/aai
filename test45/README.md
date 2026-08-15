# Test 45 — hierarchical consolidation

Same ARC-AGI protocol as **test44** (train every demo, then Fit / Train / Eval),
plus a second phase: **keep the nets whose pixels actually moved**, clone them
into one super sandwich, and **retrain that** on the same ARC demos.

Dense is in the sweep this time. Default is **all kinds × 1..15-cameral**, then
one hierarchical net **per cameral width**.

```bash
cd apps/aai/test45
go run . -camerals 15
```

That is ~20 kinds × 15 widths + 15 hier retrains. Short smoke:

```bash
go run . -n 5 -camerals 2 -layers dense,residual -item-time 20ms
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-n` | `0` (all) | cap tasks per split |
| `-camerals` | `15` | max hemispheres (`cam-min`..N) |
| `-cam-min` | `1` | first cameral count (1 = single mid) |
| `-only` | `0` | exactly N hemispheres — no sweep |
| `-layers` | `all` | every kind **including Dense**. `except-dense` skips it |
| `-item-time` | `125ms` | TrainStackMSE budget per demo |
| `-keep-min` | `6.0` | TrainPix floor to survive plumbing (dead cluster is ~5%) |
| `-keep-top` | `8` | max kinds kept per cameral width |
| `-skip-hier` | false | individuals only |
| `-workers` | NumCPU | concurrent **individual** jobs (hier is sequential per n) |
| `-set` | `agi1` | `agi1` \| `agi2` |

Writes `test45_results.json` (gitignored).

---

## Phase 1 — individuals

Pad every grid to 30×30 → **902-d** vector. Each job is one Sandwich:

| Arch | Sandwich |
|------|----------|
| 1-cameral | `Dense 902→H → one mid Op → Dense H→902` |
| *n*-cameral | `Dense 902→H → Hemispheres(n, add) → Dense H→902` |

A cameral is not another sequential layer. Stem/head stay Dense adapters.
`-layers cnn2` / `mha` / `residual` / … swaps the hemisphere kind.

---

## Phase 2 — plumbing (per cameral width)

After every kind at width *n* has trained:

1. **Filter.** Keep nets with `TrainPix ≥ keep-min`, capped at `keep-top`.
   If nobody clears the floor, take the top 2 so a merge still exists.
2. **Clone.** Copy `dna.CollectStores` + GDN blobs into fresh sandwiches
   (same geometry as the survivor).
3. **Family tree.** Group survivor mids:
   - **dense-like:** dense, residual, sequential, metacognition, swiglu
   - **conv:** cnn1/2/3, convt1/2/3
   - **seq:** mha, lstm, rnn, mamba, gdn
   - **norm:** layernorm, rmsnorm, softmax, kmeans
4. **Inner merge:** `CombineAvg` inside a family.
5. **Outer merge:** `CombineFilter` MoE gate across families. Gate bias is
   seeded from family mean TrainPix (higher TrainPix → more initial logit).
   Stem/head come from the best TrainPix survivor so the 902-d adapters stay
   aligned.
6. **Retrain** that super sandwich on the same ARC demos (`TrainStackMSE`),
   then Fit / Train / Eval like the individuals.

This is still a 64-d hidden sandwich, not a 30×30 CNN. Consolidation is
**weight plumbing + a filter gate**, then more gradient steps on ARC.

---

## Tables at the end

1. **Per layer** — that kind at every cam size
2. **Per cam size** — every layer **and** the hier net at that width
3. **COMPARE hierarchical consolidations** — `hier/1-cameral` … `hier/15-cameral`
4. **COMPARE all** — individuals + every hier

Same Lucy columns as test44 (SoftAcc / AdaptPct / Availability / Score).
Lucy Score is still duty-cycle-ish; **TrainPix / EvalPix / exact solves**
are the ARC numbers that matter.
