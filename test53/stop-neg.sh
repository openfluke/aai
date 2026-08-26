#!/usr/bin/env bash
# Stop test53 negative-LR farm (project test53-neg).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST53_PROJECT=test53-neg
export TEST53_LRS=funny-neg
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt_neg}"
export TIDE_PORT="${TIDE_PORT:-8081}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.neg.yml
exec "$DIR/run-docker.sh" --stop
