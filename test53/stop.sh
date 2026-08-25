#!/usr/bin/env bash
# Stop test53 compose project (container + published Tide port).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53

export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"

if docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
elif command -v sg >/dev/null 2>&1 && sg docker -c 'docker info' >/dev/null 2>&1; then
  dc() { sg docker -c "docker compose --project-name $PROJECT $*"; }
elif podman compose version >/dev/null 2>&1; then
  systemctl --user start podman.socket 2>/dev/null || true
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { podman compose --project-name "$PROJECT" "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
else
  echo "error: need docker compose or podman compose" >&2
  exit 1
fi

dc down
echo "stopped · project=$PROJECT"
echo "  ckpt kept at ${TEST53_CKPT_HOST:-$DIR/test53_ckpt}/"
echo "  (safe to ./run-docker.sh --build — data is on the host, not in the image)"
