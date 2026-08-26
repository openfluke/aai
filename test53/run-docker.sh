#!/usr/bin/env bash
# test53 — compile Go binary, then put ONLY the binary in Docker (tiny context).
#
#   ./run-docker.sh              # start existing image
#   ./run-docker.sh --build      # go build (in golang container) → compose up
#   ./run-docker.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53
BIN_DIR="$DIR/.bin"
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"
mkdir -p "$TEST53_CKPT_HOST" "$BIN_DIR"

WELVET="$(cd "$DIR/../../.." && pwd)"
ROOT="$(cd "$WELVET/.." && pwd)"
if [[ ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings tide/ webgpu/ next to welvet/ under $ROOT" >&2
  exit 1
fi

# Prefer Colima's official env (fixes "already running" but docker info fails).
if command -v colima >/dev/null 2>&1; then
  if colima status 2>/dev/null | grep -qi 'Running'; then
    # shellcheck disable=SC1090
    eval "$(colima docker-env 2>/dev/null)" || true
  fi
fi

# --- compose + docker run ---
if command -v docker-compose >/dev/null 2>&1; then
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
  ENGINE=compose-bin
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
  ENGINE=docker
elif command -v sg >/dev/null 2>&1 && sg docker -c 'docker compose version' >/dev/null 2>&1; then
  dc() { sg docker -c "docker compose --project-name $PROJECT $*"; }
  ENGINE=docker-sg
elif podman compose version >/dev/null 2>&1; then
  systemctl --user start podman.socket 2>/dev/null || true
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { podman compose --project-name "$PROJECT" "$@"; }
  ENGINE=podman
else
  echo "error: need docker-compose or docker compose" >&2
  exit 1
fi

drun() {
  if [[ "$ENGINE" == docker-sg ]]; then
    sg docker -c "docker run $*"
  elif [[ "$ENGINE" == podman* ]]; then
    podman run "$@"
  else
    docker run "$@"
  fi
}

if [[ "$ENGINE" != podman* ]]; then
  if ! docker info >/dev/null 2>&1; then
    echo "error: docker not reachable for compose" >&2
    echo "  try:  eval \"\$(colima docker-env)\" && docker info" >&2
    echo "  or:   colima restart" >&2
    docker info 2>&1 | head -5 >&2 || true
    exit 1
  fi
fi

compile_binary() {
  echo "compiling linux binary via golang container (bind mounts)…"
  mkdir -p "$BIN_DIR"
  cp -f "$DIR/Dockerfile" "$BIN_DIR/Dockerfile"
  cp -f "$DIR/.env.example" "$BIN_DIR/.env.example"

  local platform=""
  platform="$(docker version -f '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null || true)"

  run_compile() {
    drun --rm "$@" \
      -v "$WELVET:/src/welvet:ro" \
      -v "$ROOT/tide:/src/tide:ro" \
      -v "$ROOT/webgpu:/src/webgpu:ro" \
      -v "$BIN_DIR:/out" \
      -w /src/welvet/apps/aai/test53 \
      -e CGO_ENABLED=1 \
      -e GOOS=linux \
      -e GOFLAGS=-mod=readonly \
      golang:1.22-bookworm \
      bash -ec '
        apt-get update -qq
        apt-get install -y -qq gcc libc6-dev >/dev/null
        go build -buildvcs=false -trimpath -ldflags="-s -w" -o /out/test53 .
        ls -lh /out/test53
      '
  }
  if [[ -n "$platform" && "$platform" == */* ]]; then
    run_compile --platform "$platform"
  else
    run_compile
  fi

  if [[ ! -f "$BIN_DIR/test53" ]]; then
    echo "error: compile failed — no $BIN_DIR/test53" >&2
    exit 1
  fi
  echo "binary ready: $(du -h "$BIN_DIR/test53" | awk '{print $1}') → compose context .bin/"
}

cmd="${1:-up}"
case "$cmd" in
  up|""|--build|build)
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env"
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
      compile_binary
      dc up --build -d
    else
      if [[ ! -f "$BIN_DIR/test53" ]]; then
        echo "no .bin/test53 yet — compiling once"
        compile_binary
      fi
      dc up -d
    fi
    echo
    echo "up · project=$PROJECT  engine=$ENGINE  resume=true"
    echo "  tide  http://localhost:${TIDE_PORT:-8080}"
    echo "  ckpt  $TEST53_CKPT_HOST/"
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
    echo "Usage: $0 [up|--build|--logs|--stop|--status|--restart]" >&2
    exit 2
    ;;
esac
