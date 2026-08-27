#!/usr/bin/env bash
# Stop test53 extreme funny-LR farm. Default cam1; use ./stop-hi.sh cam3
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test53_parse_layer_args "$@"
set -- ${TEST53_REMAINING_ARGS[@]+"${TEST53_REMAINING_ARGS[@]}"}
export TEST53_LRS=funny-hi
test53_apply_cam_env hi
export TEST53_CKPT_HOST="$(test53_ckpt_host_for_layer "$(test53_ckpt_root_for_band hi "$TEST53_CAMS")" "$TEST53_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test53_tide_port "${TEST53_CAMS:-1}" hi)}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.hi.yml
echo "stopping HI cam=$TEST53_CAMS project=$TEST53_PROJECT"
exec "$DIR/run-docker.sh" --stop
