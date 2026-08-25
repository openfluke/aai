#!/usr/bin/env bash
# Stop test51 layer processes started by ./start.sh
#
# Usage:
#   ./stop.sh              # stop all layers in run/
#   ./stop.sh dense
#   ./stop.sh dense,dense-wide
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ -f "$DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DIR/.env"
  set +a
fi

RUN_DIR="${TEST51_RUN_DIR:-run}"
ALL_LAYERS=(dense dense-wide dense-deep dense-deep-wide)

stop_one() {
  local layer="$1"
  local pidfile="$RUN_DIR/$layer.pid"
  if [[ ! -f "$pidfile" ]]; then
    # legacy single-process pid
    if [[ "$layer" == "_legacy" && -f "$DIR/test51.pid" ]]; then
      pidfile="$DIR/test51.pid"
    else
      echo "· $layer not running (no pid file)"
      return 0
    fi
  fi
  local pid
  pid="$(cat "$pidfile")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "· $layer stale pid $pid — removed"
    rm -f "$pidfile"
    return 0
  fi
  echo "· stopping $layer (pid $pid)..."
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  still up — SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
  echo "  stopped"
}

TARGETS=()
if [[ $# -eq 0 ]]; then
  # stop every pid under run/ + legacy
  if [[ -d "$RUN_DIR" ]]; then
    for f in "$RUN_DIR"/*.pid; do
      [[ -e "$f" ]] || continue
      base="$(basename "$f" .pid)"
      TARGETS+=("$base")
    done
  fi
  if [[ -f "$DIR/test51.pid" ]]; then
    TARGETS+=("_legacy")
  fi
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "nothing running"
    exit 0
  fi
else
  IFS=',' read -ra RAW <<<"$*"
  for tok in "${RAW[@]}"; do
    tok="$(echo "$tok" | tr '[:upper:]' '[:lower:]' | xargs)"
    [[ -z "$tok" ]] && continue
    if [[ "$tok" == "all" ]]; then
      TARGETS=("${ALL_LAYERS[@]}")
      break
    fi
    TARGETS+=("$tok")
  done
fi

for layer in "${TARGETS[@]}"; do
  stop_one "$layer"
done

# drop manifest if nothing left
left=0
if [[ -d "$RUN_DIR" ]]; then
  for f in "$RUN_DIR"/*.pid; do
    [[ -e "$f" ]] || continue
    left=1
    break
  done
fi
if [[ "$left" -eq 0 ]]; then
  rm -f "$RUN_DIR/manifest.txt"
fi
echo "done"
