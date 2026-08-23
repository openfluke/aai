#!/usr/bin/env bash
# Start test51 in the background (survives SSH logout). Logs → test51.log
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PIDFILE="$DIR/test51.pid"
LOGFILE="$DIR/test51.log"
BIN="$DIR/test51"

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "test51 already running (pid $pid)"
    echo "dash: http://${ip:-<host-ip>}:5151"
    echo "log:  tail -f $LOGFILE"
    exit 0
  fi
  rm -f "$PIDFILE"
fi

echo "building..."
go build -o "$BIN" .

echo "starting in background..."
nohup "$BIN" "$@" >>"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

sleep 1
if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "failed to start — check log:"
  tail -20 "$LOGFILE" 2>/dev/null || true
  rm -f "$PIDFILE"
  exit 1
fi

ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "started pid $(cat "$PIDFILE")"
echo "log:  tail -f $LOGFILE"
echo "dash: http://${ip:-<host-ip>}:5151  (open in browser → Start)"
