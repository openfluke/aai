#!/usr/bin/env bash
# test53 negative-LR farm (−100m … −0.02). Own ckpt + Tide :8081.
#
#   ./run-docker-neg.sh --build
#   ./run-docker-neg.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export TEST53_PROJECT=test53-neg
export TEST53_LRS="${TEST53_LRS:-funny-neg}"
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt_neg}"
export TIDE_PORT="${TIDE_PORT:-8081}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.neg.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 NEG · lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
