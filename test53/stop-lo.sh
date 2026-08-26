#!/usr/bin/env bash
# Stop test53 mild funny-LR farm (project test53-lo).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST53_PROJECT=test53-lo
export TEST53_LRS=funny-lo
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"
export TIDE_PORT="${TIDE_PORT:-8080}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.lo.yml
exec "$DIR/run-docker.sh" --stop
