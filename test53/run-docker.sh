#!/usr/bin/env bash
# test53 dayroute — Docker or Podman Compose. ckpt on host ./test53_ckpt
#
#   ./run-docker.sh              # start existing image (fast)
#   ./run-docker.sh --build      # slim staging ctx + rebuild + start
#   ./run-docker.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"
mkdir -p "$TEST53_CKPT_HOST"

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"

WELVET="$(cd "$DIR/../../.." && pwd)"
ROOT="$(cd "$WELVET/.." && pwd)"
if [[ ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings tide/ webgpu/ next to welvet/ under $ROOT" >&2
  exit 1
fi

if docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
  ENGINE=docker
elif command -v sg >/dev/null 2>&1 && sg docker -c 'docker info' >/dev/null 2>&1; then
  dc() { sg docker -c "DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose --project-name $PROJECT $*"; }
  ENGINE=docker-sg
elif podman compose version >/dev/null 2>&1; then
  systemctl --user start podman.socket 2>/dev/null || true
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { podman compose --project-name "$PROJECT" "$@"; }
  ENGINE=podman
elif command -v docker-compose >/dev/null 2>&1; then
  dc() { DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose --project-name "$PROJECT" "$@"; }
  ENGINE=compose-bin
else
  echo "error: need docker compose or podman compose" >&2
  exit 1
fi

# Pack only what the image needs into .build-ctx/ (classic builder safe).
pack_build_ctx() {
  local stage="$DIR/.build-ctx"
  echo "packing slim build context → $stage"
  rm -rf "$stage"
  mkdir -p "$stage/welvet" "$stage/tide" "$stage/webgpu"

  # welvet: library sources + test53 only (skip other apps / site / ckpts)
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' \
      --exclude 'node_modules/' \
      --exclude 'openfluke.github.io/' \
      --exclude 'apps/' \
      --exclude '*_ckpt/' \
      --exclude '*.log' \
      --exclude '.env' \
      "$WELVET"/ "$stage/welvet/"
    mkdir -p "$stage/welvet/apps/aai"
    rsync -a \
      --exclude 'test53_ckpt/' \
      --exclude '.build-ctx/' \
      --exclude 'test53' \
      --exclude 'test53_*' \
      --exclude '.env' \
      --exclude '*.log' \
      "$DIR"/ "$stage/welvet/apps/aai/test53/"
    rsync -a --exclude '.git/' --exclude 'node_modules/' --exclude '*.log' \
      "$ROOT/tide"/ "$stage/tide/"
    rsync -a \
      --exclude '.git/' \
      --exclude 'node_modules/' \
      --exclude 'examples/' \
      --exclude 'wgpu/lib/android/' \
      --exclude 'wgpu/lib/darwin/' \
      --exclude 'wgpu/lib/ios/' \
      --exclude 'wgpu/lib/windows/' \
      "$ROOT/webgpu"/ "$stage/webgpu/"
  else
    echo "error: rsync required to pack slim context (brew install rsync)" >&2
    exit 1
  fi

  cp "$DIR/Dockerfile" "$stage/Dockerfile"
  local bytes
  bytes=$(du -sk "$stage" | awk '{print $1}')
  echo "packed ${bytes} KiB (≈ $(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1024}') MiB) — not GBs of ~/git"
}

cmd="${1:-up}"
case "$cmd" in
  up|""|--build|build)
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env (layers=all modes=all dtypes=all workers=4)"
    fi
    mkdir -p "$TEST53_CKPT_HOST"
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
    if [[ "$cmd" == "--build" || "$cmd" == "build" ]]; then
      pack_build_ctx
      dc up --build -d
    else
      if [[ ! -d .build-ctx ]]; then
        echo "no .build-ctx yet — packing once so compose has a context"
        pack_build_ctx
      fi
      dc up -d
    fi
    echo
    echo "up · project=$PROJECT  engine=$ENGINE  restart=always  resume=true"
    echo "  tide  http://localhost:${TIDE_PORT:-8080}"
    echo "  ckpt  $TEST53_CKPT_HOST/   (HOST bind — survives rebuild)"
    if [[ -f "$TEST53_CKPT_HOST/progress.json" ]]; then
      echo "  resume data present"
    fi
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
