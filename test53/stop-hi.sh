#!/usr/bin/env bash
# Stop test53 extreme funny-LR farm (project test53-hi).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST53_PROJECT=test53-hi
export TEST53_LRS=funny-hi
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt_hi}"
export TIDE_PORT="${TIDE_PORT:-8082}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.hi.yml
exec "$DIR/run-docker.sh" --stop
