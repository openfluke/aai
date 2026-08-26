#!/usr/bin/env bash
# test53 mild funny-LR farm: 0.02, 2, 200, 2000. Reuses test53_ckpt + Tide :8080.
#
#   ./run-docker-lo.sh --build
#   ./run-docker-lo.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export TEST53_PROJECT="${TEST53_PROJECT:-test53-lo}"
export TEST53_LRS="${TEST53_LRS:-funny-lo}"
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"
export TIDE_PORT="${TIDE_PORT:-8080}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.lo.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 LO · lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
