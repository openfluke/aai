#!/usr/bin/env bash
# test54 deep farm on hi band port/ckpt (same LR 0.05 — for second machine).
#
#   ./run-docker-hi.sh [cam3] [layer] --build
#   ./run-docker-hi.sh cam3 mamba --build
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}

export TEST54_LRS="${TEST54_LRS:-0.05}"
export TEST54_DURATION="${TEST54_DURATION:-15s}"
export TEST54_DEPTH="${TEST54_DEPTH:-4}"
export TEST54_HIDDEN="${TEST54_HIDDEN:-32}"
test54_apply_cam_env hi
export TEST54_CKPT_HOST="$(test54_ckpt_host_for_layer "$(test54_ckpt_root_for_band hi "$TEST54_CAMS")" "$TEST54_LAYER")"
export TIDE_PORT="${TIDE_PORT:-$(test54_tide_port "${TEST54_CAMS:-1}" hi)}"
export TEST54_COMPOSE_OVERRIDE=docker-compose.hi.yml

mkdir -p "$TEST54_CKPT_HOST"
echo "test54 HI · cam=$TEST54_CAMS · layer=$TEST54_LAYER  depth=$TEST54_DEPTH  lrs=$TEST54_LRS  dur=$TEST54_DURATION  ckpt=$TEST54_CKPT_HOST  tide=:$TIDE_PORT"
exec "$DIR/run-docker.sh" "$@"
