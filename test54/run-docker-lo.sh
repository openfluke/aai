#!/usr/bin/env bash
# test54 deep farm — lo LRs: 0.5, 5, 50, 500, 5000
# Default: cam1 + cam3 (m4 gets both Tide ports :9080 + :9100)
#
#   ./run-docker-lo.sh mamba --build              # cam1 + cam3
#   ./run-docker-lo.sh cam1 mamba --build         # cam1 only → :9080
#   ./run-docker-lo.sh cam3 mamba --build         # cam3 only → :9100
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}

export TEST54_LRS="${TEST54_LRS:-funny-lo}"
export TEST54_DURATION="${TEST54_DURATION:-15s}"
export TEST54_DEPTH="${TEST54_DEPTH:-4}"
export TEST54_HIDDEN="${TEST54_HIDDEN:-32}"

test54_run_for_cams lo docker-compose.lo.yml "$@"
