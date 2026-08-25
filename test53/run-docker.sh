#!/usr/bin/env bash
# test53 dayroute in Docker — restart: always, ckpt on host ./test53_ckpt
#
#   ./run-docker.sh
#   ./run-docker.sh --logs
#   ./run-docker.sh --stop
#   ./run-docker.sh --status
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53

ROOT="$(cd "$DIR/../../../.." && pwd)"
if [[ ! -d "$ROOT/welvet" || ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings welvet/ tide/ webgpu/ under $ROOT" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
else
  echo "error: need 'docker compose' or 'docker-compose'" >&2
  exit 1
fi

cmd="${1:-up}"
case "$cmd" in
  up|"")
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env (layers=all modes=all dtypes=all workers=4)"
    fi
    mkdir -p test53_ckpt
    if command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -tiTCP:8080 -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        echo "killing host listeners on :8080 → $pids"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
      fi
    fi
    dc up --build -d
    echo
    echo "up · project=$PROJECT  restart=always  resume=true"
    echo "  tide  http://localhost:${TIDE_PORT:-8080}"
    echo "  ckpt  $DIR/test53_ckpt/   (HOST — not inside the container)"
    echo "  logs  ./run-docker.sh --logs"
    ;;
  --logs|logs)
    dc logs -f --tail=80
    ;;
  --stop|stop)
    dc down
    ;;
  --status|status)
    dc ps
    echo
    dc logs --tail=30
    ;;
  --restart|restart)
    dc restart
    ;;
  *)
    echo "Usage: $0 [up|--logs|--stop|--status|--restart]" >&2
    exit 2
    ;;
esac
