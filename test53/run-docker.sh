#!/usr/bin/env bash
# test53 dayroute — Docker or Podman Compose. ckpt on host ./test53_ckpt
#
#   ./run-docker.sh              # start existing image (fast)
#   ./run-docker.sh --build      # rebuild image then start
#   ./run-docker.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53

ROOT="$(cd "$DIR/../../../.." && pwd)"
if [[ ! -d "$ROOT/welvet" || ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings welvet/ tide/ webgpu/ under $ROOT" >&2
  exit 1
fi

# Prefer real Docker when the daemon is reachable; else Podman (Fedora).
if docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
  ENGINE=docker
elif podman compose version >/dev/null 2>&1; then
  systemctl --user start podman.socket 2>/dev/null || true
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { podman compose --project-name "$PROJECT" "$@"; }
  ENGINE=podman
elif command -v docker-compose >/dev/null 2>&1; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
  ENGINE=compose-bin
else
  echo "error: need docker compose or podman compose" >&2
  echo "  Fedora:  sudo dnf install -y moby-engine docker-cli docker-compose docker-compose-switch" >&2
  echo "       then: sudo systemctl enable --now docker && sudo usermod -aG docker \$USER" >&2
  exit 1
fi

cmd="${1:-up}"
case "$cmd" in
  up|""|--build|build)
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env (layers=all modes=all dtypes=all workers=4)"
    fi
    mkdir -p test53_ckpt
    # Rootless podman socket (no-op if docker engine is up).
    if [[ "$ENGINE" == podman* ]]; then
      systemctl --user start podman.socket 2>/dev/null || true
    fi
    if command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -tiTCP:8080 -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        echo "killing host listeners on :8080 → $pids"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
      fi
    fi
    echo "engine=$ENGINE"
    # Default: start existing image. Pass --build only when source changed.
    if [[ "$cmd" == "--build" || "$cmd" == "build" ]]; then
      dc up --build -d
    else
      dc up -d
    fi
    echo
    echo "up · project=$PROJECT  engine=$ENGINE  restart=always  resume=true"
    echo "  tide  http://localhost:${TIDE_PORT:-8080}"
    echo "  ckpt  $DIR/test53_ckpt/   (HOST — not inside the container)"
    echo "  logs  ./run-docker.sh --logs"
    echo "  rebuild  ./run-docker.sh --build"
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
    echo "Usage: $0 [up|--build|--logs|--stop|--status|--restart]" >&2
    exit 2
    ;;
esac
