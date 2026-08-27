#!/usr/bin/env bash
# List all test53 compose projects (cam × band). Does not stop anything.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=layer-env.sh
source "$DIR/layer-env.sh"

dc_for() {
  local project="$1" override="$2"
  if podman compose version >/dev/null 2>&1; then
    podman compose --project-name "$project" -f "$DIR/docker-compose.yml" -f "$DIR/$override" ps
  elif command -v podman-compose >/dev/null 2>&1; then
    podman-compose --project-name "$project" -f "$DIR/docker-compose.yml" -f "$DIR/$override" ps
  else
    echo "(need podman compose)"
  fi
}

echo "test53 farms (each cam×band is a separate compose project):"
for cam in 1 3; do
  for band in lo hi; do
    TEST53_CAMS=$cam
    test53_apply_cam_env "$band"
    case "$band" in
      lo) o=docker-compose.lo.yml ;;
      hi) o=docker-compose.hi.yml ;;
    esac
    echo "--- cam${cam}-${band} · project=${TEST53_PROJECT} · port=$(test53_tide_port "$cam" "$band")"
    dc_for "$TEST53_PROJECT" "$o" 2>/dev/null || true
    echo
  done
done
