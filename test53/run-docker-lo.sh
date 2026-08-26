#!/usr/bin/env bash
# test53 mild funny-LR farm: 0.02, 2, 200, 2000. One layer → own ckpt subfolder.
#
#   ./run-docker-lo.sh [layer] --build
#   ./run-docker-lo.sh convt2 --build
#   ./run-docker-lo.sh --logs
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test53_parse_layer_args "$@"
set -- "${TEST53_REMAINING_ARGS[@]}"

export TEST53_PROJECT="${TEST53_PROJECT:-test53-lo}"
export TEST53_LRS="${TEST53_LRS:-funny-lo}"
export TEST53_CKPT_HOST="$(test53_ckpt_host_for_layer test53_ckpt "$TEST53_LAYER")"
export TIDE_PORT="${TIDE_PORT:-8080}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.lo.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 LO · layer=$TEST53_LAYER  lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
