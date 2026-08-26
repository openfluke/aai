#!/usr/bin/env bash
# Default dayroute LPD layers (run one at a time).
cat <<'EOF'
Run one layer at a time — each gets its own ckpt subfolder:

  ./run-docker-lo.sh dense --build
  ./run-docker-lo.sh convt2 --build
  ./stop-lo.sh
  ./run-docker-lo.sh mha --build

Layers (16):
  dense
  cnn1 cnn2 cnn3
  convt1 convt2 convt3
  mha lstm rnn mamba gdn swiglu residual
  layernorm rmsnorm

~4 LRs × 29 modes × 34 dtypes ≈ 3.9k jobs per layer (lo or hi half).
EOF
