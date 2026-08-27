#!/usr/bin/env bash
# test53 negative-LR farm (−100m … −0.02). One layer → own ckpt subfolder.
#
#   ./run-docker-neg.sh [layer] --build
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test53_parse_layer_args "$@"
set -- ${TEST53_REMAINING_ARGS[@]+"${TEST53_REMAINING_ARGS[@]}"}

export TEST53_LRS="${TEST53_LRS:-funny-neg}"
test53_apply_cam_env neg
export TEST53_CKPT_HOST="$(test53_ckpt_host_for_layer "$(test53_ckpt_root_for_band neg "$TEST53_CAMS")" "$TEST53_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test53_tide_port "${TEST53_CAMS:-1}" neg)}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.neg.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 NEG · cam=$TEST53_CAMS · layer=$TEST53_LAYER  lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
