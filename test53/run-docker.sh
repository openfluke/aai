#!/usr/bin/env bash
# test53 — compile Go binary, then put ONLY the binary in Docker (tiny context).
#
#   ./run-docker.sh              # start existing image
#   ./run-docker.sh --build      # go build (in golang container) → docker up
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

docker_ok() { docker info >/dev/null 2>&1; }

if docker_ok && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
  drun() { docker run "$@"; }
  ENGINE=docker
elif command -v sg >/dev/null 2>&1 && sg docker -c 'docker info' >/dev/null 2>&1; then
  dc() { sg docker -c "docker compose --project-name $PROJECT $*"; }
  drun() { sg docker -c "docker run $*"; }
  ENGINE=docker-sg
elif podman compose version >/dev/null 2>&1; then
  systemctl --user start podman.socket 2>/dev/null || true
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { podman compose --project-name "$PROJECT" "$@"; }
  drun() { podman run "$@"; }
  ENGINE=podman
elif command -v docker-compose >/dev/null 2>&1 && docker_ok; then
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
  drun() { docker run "$@"; }
  ENGINE=compose-bin
else
  echo "error: need docker (or podman) compose" >&2
  exit 1
fi

# Compile linux binary inside golang image; sources are BIND-MOUNTED (not tarred
# as build context). Output lands in .bin/test53 — that's all Docker ever sees.
compile_binary() {
  echo "compiling linux binary via golang container (bind mounts, no GB context)…"
  mkdir -p "$BIN_DIR"
  # Dockerfile for the runtime image lives next to the binary in .bin/
  cp -f "$DIR/Dockerfile" "$BIN_DIR/Dockerfile"
  cp -f "$DIR/.env.example" "$BIN_DIR/.env.example"

  # Match container platform so the binary runs in debian slim.
  local platform=""
  if docker_ok || [[ "$ENGINE" == docker* ]]; then
    platform="$(docker version -f '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null || true)"
  fi
  local plat_args=()
  if [[ -n "$platform" && "$platform" == */* ]]; then
    plat_args=(--platform "$platform")
  fi

  # Layout inside builder matches go.mod replace paths:
  #   /src/welvet/apps/aai/test53  +  /src/tide  +  /src/webgpu
  drun --rm \
    "${plat_args[@]}" \
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
      # Do NOT go mod tidy — welvet is mounted :ro. go.sum must already be committed.
      go build -trimpath -ldflags="-s -w" -o /out/test53 .
      ls -lh /out/test53
    '

  if [[ ! -x "$BIN_DIR/test53" && ! -f "$BIN_DIR/test53" ]]; then
    echo "error: compile failed — no $BIN_DIR/test53" >&2
    exit 1
  fi
  echo "binary ready: $(du -h "$BIN_DIR/test53" | awk '{print $1}') → docker context is just .bin/"
}

cmd="${1:-up}"
case "$cmd" in
  up|""|--build|build)
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env"
    fi
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
