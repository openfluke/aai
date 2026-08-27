#!/usr/bin/env bash
# test53 extreme funny-LR farm: 20000, 1m, 10m, 100m. One layer → own ckpt subfolder.
#
#   ./run-docker-hi.sh [layer] --build
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test53_parse_layer_args "$@"
set -- "${TEST53_REMAINING_ARGS[@]}"

export TEST53_PROJECT="${TEST53_PROJECT:-test53-hi}"
export TEST53_LRS="${TEST53_LRS:-funny-hi}"
test53_apply_cam_env hi
export TEST53_CKPT_HOST="$(test53_ckpt_host_for_layer "$(test53_ckpt_root_for_band hi "$TEST53_CAMS")" "$TEST53_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test53_tide_port "${TEST53_CAMS:-1}" hi)}"
export TEST53_COMPOSE_OVERRIDE=docker-compose.hi.yml

mkdir -p "$TEST53_CKPT_HOST"
echo "test53 HI · cam=$TEST53_CAMS · layer=$TEST53_LAYER  lrs=$TEST53_LRS  ckpt=$TEST53_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
