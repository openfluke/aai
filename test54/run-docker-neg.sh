#!/usr/bin/env bash
# test54 negative-LR farm (−100m … −0.02). One layer → own ckpt subfolder.
#
#   ./run-docker-neg.sh [layer] --build
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}

export TEST54_LRS="${TEST54_LRS:-funny-neg}"
test54_apply_cam_env neg
export TEST54_CKPT_HOST="$(test54_ckpt_host_for_layer "$(test54_ckpt_root_for_band neg "$TEST54_CAMS")" "$TEST54_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test54_tide_port "${TEST54_CAMS:-1}" neg)}"
export TEST54_COMPOSE_OVERRIDE=docker-compose.neg.yml

mkdir -p "$TEST54_CKPT_HOST"
echo "test54 NEG · cam=$TEST54_CAMS · layer=$TEST54_LAYER  lrs=$TEST54_LRS  ckpt=$TEST54_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
