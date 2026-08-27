#!/usr/bin/env bash
# test53 — Podman-first (Mac + Linux). Compiles Go binary, then compose up.
#
#   ./run-docker.sh              # start existing image
#   ./run-docker.sh --build      # go build (podman) → compose up
#   ./run-docker.sh --logs|--stop|--status|--restart
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT="${TEST53_PROJECT:-test53}"
BIN_DIR="$DIR/.bin"
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"
export TEST53_LRS="${TEST53_LRS:-funny-lo}"
export TIDE_PORT="${TIDE_PORT:-8080}"
mkdir -p "$TEST53_CKPT_HOST" "$BIN_DIR"

WELVET="$(cd "$DIR/../../.." && pwd)"
ROOT="$(cd "$WELVET/.." && pwd)"
if [[ ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings tide/ webgpu/ next to welvet/ under $ROOT" >&2
  exit 1
fi

is_mac() { [[ "$(uname -s)" == Darwin ]]; }

ensure_podman() {
  if command -v podman >/dev/null 2>&1; then
    return 0
  fi
  if ! is_mac; then
    echo "error: podman not installed" >&2
    echo "  Fedora: sudo dnf install -y podman podman-compose" >&2
    exit 1
  fi
  if ! command -v brew >/dev/null 2>&1; then
    echo "error: podman missing and Homebrew not found — install brew, then: brew install podman" >&2
    exit 1
  fi
  echo "podman not found — installing via Homebrew…"
  brew install podman
}

ensure_podman_machine() {
  is_mac || return 0
  # Init default machine if none exists.
  if ! podman machine list --format '{{.Name}}' 2>/dev/null | grep -q .; then
    echo "podman machine init (first time)…"
    podman machine init --cpus 4 --memory 8192 --disk-size 60
  fi
  start_podman_machine() {
    echo "podman machine start…"
    podman machine start
  }
  if ! podman info >/dev/null 2>&1; then
    start_podman_machine
  fi
  if ! podman info >/dev/null 2>&1; then
    echo "podman still unreachable — restarting machine…" >&2
    podman machine stop 2>/dev/null || true
    start_podman_machine
  fi
  if ! podman info >/dev/null 2>&1; then
    echo "error: podman machine not reachable (try: podman machine start && podman info)" >&2
    exit 1
  fi
}

ensure_podman_compose() {
  if podman compose version >/dev/null 2>&1; then
    return 0
  fi
  if command -v podman-compose >/dev/null 2>&1; then
    return 0
  fi
  if is_mac && command -v brew >/dev/null 2>&1; then
    echo "installing docker-compose plugin / podman-compose…"
    brew install docker-compose 2>/dev/null || brew install podman-compose || true
  fi
  if podman compose version >/dev/null 2>&1; then
    return 0
  fi
  if command -v podman-compose >/dev/null 2>&1; then
    return 0
  fi
  echo "error: need 'podman compose' or 'podman-compose'" >&2
  exit 1
}

ensure_podman
# Don't inherit a broken Colima DOCKER_HOST — podman has its own machine.
unset DOCKER_HOST || true
ensure_podman_machine
ensure_podman_compose

compose_args=(-f "$DIR/docker-compose.yml")
if [[ -n "${TEST53_COMPOSE_OVERRIDE:-}" ]]; then
  compose_args+=(-f "$DIR/$TEST53_COMPOSE_OVERRIDE")
fi

if podman compose version >/dev/null 2>&1; then
  dc() { podman compose --project-name "$PROJECT" "${compose_args[@]}" "$@"; }
  ENGINE=podman-compose
else
  dc() { podman-compose --project-name "$PROJECT" "${compose_args[@]}" "$@"; }
  ENGINE=podman-compose-bin
fi
drun() { podman run "$@"; }

compile_binary() {
  echo "compiling linux binary via podman golang container…"
  mkdir -p "$BIN_DIR"
  cp -f "$DIR/Dockerfile" "$BIN_DIR/Dockerfile"
  cp -f "$DIR/.env.example" "$BIN_DIR/.env.example"

  local platform=""
  platform="$(podman info -f '{{.Host.OS}}/{{.Host.Arch}}' 2>/dev/null || true)"
  # Normalize arch names podman may report.
  platform="${platform/arm64/arm64}"
  platform="${platform/aarch64/arm64}"
  platform="${platform/x86_64/amd64}"

  run_compile() {
    drun --rm "$@" \
      -v "$WELVET:/src/welvet:ro" \
      -v "$ROOT/tide:/src/tide:ro" \
      -v "$ROOT/webgpu:/src/webgpu:ro" \
      -v "$BIN_DIR:/out:Z" \
      -w /src/welvet/apps/aai/test53 \
      -e CGO_ENABLED=1 \
      -e GOOS=linux \
      -e GOFLAGS=-mod=readonly \
      docker.io/library/golang:1.22-bookworm \
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
    if command -v lsof >/dev/null 2>&1 && [[ "${TEST53_FORCE_PORT:-}" == "1" ]]; then
      pids=$(lsof -tiTCP:"${TIDE_PORT}" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        echo "TEST53_FORCE_PORT=1 — killing host listeners on :${TIDE_PORT} → $pids"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
      fi
    elif command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -tiTCP:"${TIDE_PORT}" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        echo "warn: :${TIDE_PORT} already in use (pid $pids) — not killing (set TEST53_FORCE_PORT=1 to override)"
        echo "      another cam/band may be running; use ./status-all.sh to check"
      fi
    fi
    echo "engine=$ENGINE  project=$PROJECT  lrs=$TEST53_LRS"
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
    if [[ "$cmd" != up && "$cmd" != --* && "$cmd" != build ]]; then
      echo "error: unknown command '$cmd'" >&2
      echo "  layer names go on run-docker-lo/hi/neg.sh, not run-docker.sh:" >&2
      echo "    ./run-docker-hi.sh dense --build" >&2
      echo "  (git pull if you still see ckpt=test53_ckpt_hi without /dense/)" >&2
    fi
    echo "Usage: $0 [up|--build|--logs|--stop|--status|--restart]" >&2
    exit 2
    ;;
esac
