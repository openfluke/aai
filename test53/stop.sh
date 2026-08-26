#!/usr/bin/env bash
# Stop test53 (Podman compose project). ckpt kept on host.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test53
export TEST53_CKPT_HOST="${TEST53_CKPT_HOST:-$DIR/test53_ckpt}"

if command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
  dc() { podman compose --project-name "$PROJECT" "$@"; }
elif command -v podman-compose >/dev/null 2>&1; then
  dc() { podman-compose --project-name "$PROJECT" "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  dc() { docker-compose --project-name "$PROJECT" "$@"; }
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  dc() { docker compose --project-name "$PROJECT" "$@"; }
else
  echo "error: need podman compose (or docker compose fallback)" >&2
  exit 1
fi

dc down
echo "stopped · project=$PROJECT"
echo "  ckpt kept at $TEST53_CKPT_HOST/"
