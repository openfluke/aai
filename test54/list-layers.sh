#!/usr/bin/env bash
# Default dayroute deep layers (run one at a time).
cat <<'EOF'
test54 — deep sandwich (stem + mid×DEPTH + head). Default DEPTH=4, LR=0.05, dur=15s.

  ./run-docker-lo.sh mamba --build
  ./run-docker-lo.sh cam3 mamba --build
  ./stop-lo.sh
  ./run-docker-lo.sh cam3 lstm --build

Layers (16):
  dense
  cnn1 cnn2 cnn3
  convt1 convt2 convt3
  mha lstm rnn mamba gdn swiglu residual
  layernorm rmsnorm

~1 LR × 21 modes × 34 dtypes ≈ 714 jobs per layer.
EOF
