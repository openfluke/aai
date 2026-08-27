#!/usr/bin/env bash
# test54 deep farm on hi band port/ckpt (same LR 0.05 — for second machine).
#
#   ./run-docker-hi.sh mamba --build
#   ./run-docker-hi.sh cam3 mamba --build
#   ./run-docker-hi.sh both mamba --build
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

test54_run_for_cams hi docker-compose.hi.yml "$@"
