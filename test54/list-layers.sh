#!/usr/bin/env bash
# Default dayroute deep layers (run one at a time).
cat <<'EOF'
test54 — deep sandwich (stem + mid×DEPTH + head). Default DEPTH=4, LR=0.05, dur=15s.

  ./run-docker-lo.sh mamba --build              # cam1 :9080
  ./run-docker-lo.sh cam3 mamba --build         # cam3 :9100
  ./run-docker-lo.sh both mamba --build         # cam1 + cam3
  ./run-docker-hi.sh both mamba --build         # m5: cam1-hi + cam3-hi
  ./stop-lo.sh both
  ./status-all.sh

Layers (16):
  dense
  cnn1 cnn2 cnn3
  convt1 convt2 convt3
  mha lstm rnn mamba gdn swiglu residual
  layernorm rmsnorm

~1 LR × 21 modes × 34 dtypes ≈ 714 jobs per layer per cam.
EOF
