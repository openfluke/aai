#!/usr/bin/env bash
# Stop background test51 started by ./start.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/test51.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "test51 not running (no $PIDFILE)"
  exit 0
fi

pid="$(cat "$PIDFILE")"
if ! kill -0 "$pid" 2>/dev/null; then
  echo "stale pid $pid — removed pid file"
  rm -f "$PIDFILE"
  exit 0
fi

echo "stopping test51 (pid $pid)..."
kill "$pid" 2>/dev/null || true

for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done

if kill -0 "$pid" 2>/dev/null; then
  echo "still running — SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
fi

rm -f "$PIDFILE"
echo "stopped"
