#!/usr/bin/env bash
# Stop test54 negative-LR farm (project test54-neg).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST54_PROJECT=test54-neg
export TEST54_LRS=funny-neg
export TEST54_CKPT_HOST="${TEST54_CKPT_HOST:-$DIR/test54_ckpt_neg}"
export TIDE_PORT="${TIDE_PORT:-8081}"
export TEST54_COMPOSE_OVERRIDE=docker-compose.neg.yml
exec "$DIR/run-docker.sh" --stop
