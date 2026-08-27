#!/usr/bin/env bash
# Stop test54 lo farm(s). Default cam1; cam3 / both supported.
#   ./stop-lo.sh
#   ./stop-lo.sh cam3
#   ./stop-lo.sh both
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}
export TEST54_LRS="${TEST54_LRS:-funny-lo}"

for cam in "${TEST54_CAMS_LIST[@]}"; do
  unset TIDE_PORT TEST54_CONTAINER_NAME TEST54_PROJECT TEST54_CKPT_HOST || true
  export TEST54_CAMS="$cam"
  test54_apply_cam_env lo
  export TEST54_CKPT_HOST="$(test54_ckpt_host_for_layer "$(test54_ckpt_root_for_band lo "$cam")" "$TEST54_LAYER")"
  export TEST54_COMPOSE_OVERRIDE=docker-compose.lo.yml
  echo "stopping LO cam=$cam project=$TEST54_PROJECT"
  "$DIR/run-docker.sh" --stop
done
