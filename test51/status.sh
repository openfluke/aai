#!/usr/bin/env bash
# Show running test51 layer processes, ports, and checkpoint dirs.
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
CKPT_ROOT="${TEST51_CKPT_ROOT:-test51_ckpt}"
PORT_BASE="${TEST51_PORT_BASE:-5151}"
TIDE_BASE="${TIDE_PORT_BASE:-8080}"
ALL_LAYERS=(dense dense-wide dense-deep dense-deep-wide)

ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "test51 layers @ ${ip:-127.0.0.1}"
echo "────────────────────────────────────────────────────────────"
printf "%-18s %-8s %-7s %-7s %s\n" "LAYER" "PID" "DASH" "TIDE" "CKPT"
echo "────────────────────────────────────────────────────────────"

any=0
idx=0
for layer in "${ALL_LAYERS[@]}"; do
  pidfile="$RUN_DIR/$layer.pid"
  p51=$((PORT_BASE + idx))
  ptide=$((TIDE_BASE + idx))
  ckpt="$CKPT_ROOT/$layer"
  pid="—"
  state="stopped"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      state="running"
      any=1
    else
      pid="stale"
    fi
  fi
  printf "%-18s %-8s %-7s %-7s %s\n" "$layer" "$pid" ":$p51" ":$ptide" "$ckpt/"
  if [[ "$state" == "running" ]]; then
    printf "  dash http://%s:%s   tide http://%s:%s\n" "${ip:-127.0.0.1}" "$p51" "${ip:-127.0.0.1}" "$ptide"
  fi
  idx=$((idx + 1))
done

if [[ -f "$DIR/test51.pid" ]]; then
  pid="$(cat "$DIR/test51.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo
    echo "legacy single process pid=$pid (test51.pid)"
    any=1
  fi
fi

echo "────────────────────────────────────────────────────────────"
if [[ "$any" -eq 0 ]]; then
  echo "none running — ./start.sh -i"
else
  [[ -f "$RUN_DIR/manifest.txt" ]] && echo && cat "$RUN_DIR/manifest.txt"
fi
