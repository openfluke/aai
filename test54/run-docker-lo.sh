#!/usr/bin/env bash
# test54 deep farm — fixed LR 0.05, longer jobs, selectable layer (default mamba).
#
#   ./run-docker-lo.sh mamba --build              # cam1 → :9080
#   ./run-docker-lo.sh cam3 mamba --build         # cam3 → :9100
#   ./run-docker-lo.sh both mamba --build         # cam1 + cam3
#   ./run-docker-lo.sh cam3 lstm --build
#   ./run-docker-lo.sh --logs
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

test54_run_for_cams lo docker-compose.lo.yml "$@"
