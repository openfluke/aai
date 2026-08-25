#!/usr/bin/env bash
# Launch one of the 3 dense farm configs (full permute, autostart).
#
# On each server after clone:
#   cd welvet/apps/aai/test51   # (inside chaosglue tree so tide+webgpu siblings exist)
#   ./run-config.sh 1          # server A
#   ./run-config.sh 2          # server B
#   ./run-config.sh 3          # server C
#
# Defaults to docker compose up --build -d.
#   ./run-config.sh 1 --local     # no Docker: build + background binary
#   ./run-config.sh 1 --fg        # docker compose up --build (foreground)
#   ./run-config.sh 1 --stop      # stop this config's compose project
#   ./run-config.sh 1 --logs      # follow logs
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

usage() {
  cat <<EOF
Usage: ./run-config.sh <1|2|3> [--local|--fg|--stop|--logs|--status]

  1  dense × Lucy core + mesh basics (10 modes)
  2  dense × Split/Alt family (10 modes)
  3  dense × Mesh/Step split variants (9 modes)

Each config: dense layer, TEST51_FULL=true, TEST51_AUTOSTART=true,
own checkpoint dir test51_ckpt/configN/, ports :5151 + :8080.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

N="$1"
shift
case "$N" in
  1|2|3) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "config must be 1, 2, or 3 (got $N)" >&2; usage; exit 2 ;;
esac

CFG="$DIR/configs/$N.env"
if [[ ! -f "$CFG" ]]; then
  echo "missing $CFG" >&2
  exit 1
fi

MODE=docker
ACTION=up
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) MODE=local; shift ;;
    --fg) ACTION=fg; shift ;;
    --stop) ACTION=stop; shift ;;
    --logs) ACTION=logs; shift ;;
    --status) ACTION=status; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Activate config as .env (compose + binary LoadDotEnv).
cp "$CFG" "$DIR/.env"
set -a
# shellcheck disable=SC1091
source "$DIR/.env"
set +a

PROJECT="test51-c${N}"
CKPT="${TEST51_CKPT:-test51_ckpt/config${N}}"
mkdir -p "$CKPT" "$DIR/run"

echo "════════════════════════════════════════════════════════"
echo " test51 config $N  (project $PROJECT)"
echo " layer=dense  full=true  autostart=true"
echo " modes=${TEST51_MODES}"
echo " ckpt=$CKPT"
echo " dash=http://0.0.0.0:5151  tide=http://0.0.0.0:8080"
echo "════════════════════════════════════════════════════════"

if [[ "$MODE" == "local" ]]; then
  case "$ACTION" in
    stop)
      "$DIR/stop.sh" dense 2>/dev/null || true
      if [[ -f "$DIR/run/config${N}.pid" ]]; then
        pid="$(cat "$DIR/run/config${N}.pid")"
        kill "$pid" 2>/dev/null || true
        rm -f "$DIR/run/config${N}.pid"
      fi
      echo "stopped local config $N"
      exit 0
      ;;
    logs)
      exec tail -f "$DIR/run/config${N}.log"
      ;;
    status)
      "$DIR/status.sh" || true
      exit 0
      ;;
  esac
  "$DIR/build.sh"
  # One process: dense + this mode set (ports from .env, not multi-layer offsets).
  if [[ -f "$DIR/run/config${N}.pid" ]] && kill -0 "$(cat "$DIR/run/config${N}.pid")" 2>/dev/null; then
    echo "already running pid $(cat "$DIR/run/config${N}.pid")"
    exit 0
  fi
  nohup "$DIR/test51" >>"$DIR/run/config${N}.log" 2>&1 &
  echo $! >"$DIR/run/config${N}.pid"
  sleep 1
  if ! kill -0 "$(cat "$DIR/run/config${N}.pid")" 2>/dev/null; then
    echo "failed — tail run/config${N}.log:" >&2
    tail -40 "$DIR/run/config${N}.log" || true
    exit 1
  fi
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo "started pid $(cat "$DIR/run/config${N}.pid") (autostart — training begins immediately)"
  echo "dash http://${ip:-127.0.0.1}:5151"
  echo "tide http://${ip:-127.0.0.1}:8080"
  echo "log  tail -f run/config${N}.log"
  exit 0
fi

# Docker path — build context is chaosglue (welvet + tide + webgpu siblings).
CHAOS="$(cd "$DIR/../../../.." && pwd)"
if [[ ! -d "$CHAOS/welvet" || ! -d "$CHAOS/tide" || ! -d "$CHAOS/webgpu" ]]; then
  echo "docker build needs chaosglue layout:" >&2
  echo "  $CHAOS/{welvet,tide,webgpu}" >&2
  echo "clone those three as siblings, or use: ./run-config.sh $N --local" >&2
  exit 1
fi

export TEST51_CONFIG="$N"
export TEST51_PORT="${TEST51_PORT:-5151}"
export TIDE_PORT="${TIDE_PORT:-8080}"
export TEST51_AUTOSTART=true

case "$ACTION" in
  stop)
    docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" down
    echo "stopped $PROJECT"
    ;;
  logs)
    exec docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" logs -f
    ;;
  status)
    docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" ps
    ;;
  fg)
    docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" up --build
    ;;
  up)
    docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" up --build -d
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo
    echo "up — autostart on; no Start click needed"
    echo "dash http://${ip:-127.0.0.1}:${TEST51_PORT}"
    echo "tide http://${ip:-127.0.0.1}:${TIDE_PORT}"
    echo "logs ./run-config.sh $N --logs"
    echo "stop ./run-config.sh $N --stop"
    ;;
esac
