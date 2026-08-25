#!/usr/bin/env bash
# Compile a Linux binary for test51 (native arch by default).
# Usage:
#   ./build.sh              # → ./test51 for this machine
#   ./build.sh amd64        # cross-compile GOARCH=amd64
#   ./build.sh arm64        # cross-compile GOARCH=arm64
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

ARCH="${1:-$(uname -m)}"
case "$ARCH" in
  x86_64|amd64) GOARCH=amd64 ;;
  aarch64|arm64) GOARCH=arm64 ;;
  *) GOARCH="$ARCH" ;;
esac

OUT="${OUT:-$DIR/test51}"
echo "building linux/$GOARCH → $OUT"
# Keep CGO default (webgpu stubs need it on some platforms). Override with CGO_ENABLED=0 if you know you can.
GOOS=linux GOARCH="$GOARCH" go build -trimpath -ldflags="-s -w" -o "$OUT" .
ls -lh "$OUT"
echo "ok"
