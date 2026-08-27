#!/usr/bin/env bash
# Stop test54 hi farm. Default cam1; use ./stop-hi.sh cam3
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}
export TEST54_LRS="${TEST54_LRS:-0.05}"
test54_apply_cam_env hi
export TEST54_CKPT_HOST="$(test54_ckpt_host_for_layer "$(test54_ckpt_root_for_band hi "$TEST54_CAMS")" "$TEST54_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test54_tide_port "${TEST54_CAMS:-1}" hi)}"
export TEST54_COMPOSE_OVERRIDE=docker-compose.hi.yml
echo "stopping HI cam=$TEST54_CAMS project=$TEST54_PROJECT"
exec "$DIR/run-docker.sh" --stop
