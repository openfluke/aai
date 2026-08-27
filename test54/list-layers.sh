#!/usr/bin/env bash
# Default dayroute deep layers (run one at a time).
cat <<'EOF'
test54 — deep sandwich (stem + mid×DEPTH + head). Default DEPTH=4, dur=15s.
Modes: sgd, [T][S]Sparse, Step[T][S]Sparse, Mesh[T][S]Sparse (4).
Lo LRs: 0.5, 5, 50, 500, 5000 · Hi LRs: 500k, 5m, 50m, 100m

  ./run-docker-lo.sh mamba --build              # m4: cam1 :9080 + cam3 :9100
  ./run-docker-hi.sh mamba --build              # m5: cam1-hi :9082 + cam3-hi :9102
  ./stop-lo.sh / ./stop-hi.sh                   # both cams by default
  ./status-all.sh

Layers (16):
  dense
  cnn1 cnn2 cnn3
  convt1 convt2 convt3
  mha lstm rnn mamba gdn swiglu residual
  layernorm rmsnorm

~lo 680 / hi 544 jobs per layer per cam (×2 cams default).
EOF
