#!/usr/bin/env bash
# test53 extreme funny-LR farm: 20000, 1m, 10m, 100m. Own ckpt + Tide :8082.
#
#   ./run-docker-hi.sh --build
#   ./run-docker-hi.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export TEST53_PROJECT="${TEST53_PROJECT:-test53-hi}"
export TEST53_LRS="${TEST53_LRS:-funny-hi}"
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt_hi}"
export TIDE_PORT="${TIDE_PORT:-8082}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.hi.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 HI · lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
