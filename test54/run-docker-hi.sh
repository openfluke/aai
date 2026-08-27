#!/usr/bin/env bash
# test54 deep farm — hi LRs: 500k, 5m, 50m, 100m
# Default: cam1 + cam3 (m5 gets both Tide ports :9082 + :9102)
#
#   ./run-docker-hi.sh mamba --build              # cam1-hi + cam3-hi
#   ./run-docker-hi.sh cam1 mamba --build         # cam1-hi only → :9082
#   ./run-docker-hi.sh cam3 mamba --build         # cam3-hi only → :9102
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"
test54_parse_layer_args "$@"
set -- ${TEST54_REMAINING_ARGS[@]+"${TEST54_REMAINING_ARGS[@]}"}

export TEST54_LRS="${TEST54_LRS:-funny-hi}"
export TEST54_DURATION="${TEST54_DURATION:-15s}"
export TEST54_DEPTH="${TEST54_DEPTH:-4}"
export TEST54_HIDDEN="${TEST54_HIDDEN:-32}"

test54_run_for_cams hi docker-compose.hi.yml "$@"
