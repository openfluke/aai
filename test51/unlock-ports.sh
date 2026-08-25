#!/usr/bin/env bash
# Open test51 / Tide dashboard ports on Fedora (firewalld).
#
# Usage:
#   ./unlock-ports.sh              # :5151 + :8080 (farm configs)
#   ./unlock-ports.sh --layers     # also :5152-5154 + :8081-8083 (multi-layer start.sh)
#   ./unlock-ports.sh --status     # show whether rules exist
#   ./unlock-ports.sh --lock       # remove the rules again
#
# Needs sudo for firewall-cmd. Safe to re-run (idempotent).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ -f "$DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DIR/.env"
  set +a
fi

PORT_BASE="${TEST51_PORT_BASE:-5151}"
TIDE_BASE="${TIDE_PORT_BASE:-8080}"
ZONE="${FIREWALL_ZONE:-}"

ACTION=unlock
LAYERS=0

usage() {
  cat <<EOF
Usage: ./unlock-ports.sh [--layers] [--lock|--status]

  (default)  open dash+tide base ports for farm configs
  --layers   also open +1..+3 (dense-wide / deep / deep-wide)
  --lock     close the ports again
  --status   list matching firewalld rules

Ports opened by default:
  TCP ${PORT_BASE}   test51 dash
  TCP ${TIDE_BASE}   Tide Lucy dash
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layers|-l) LAYERS=1; shift ;;
    --lock) ACTION=lock; shift ;;
    --status) ACTION=status; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ports=("${PORT_BASE}/tcp" "${TIDE_BASE}/tcp")
if [[ "$LAYERS" -eq 1 ]]; then
  for i in 1 2 3; do
    ports+=("$((PORT_BASE + i))/tcp" "$((TIDE_BASE + i))/tcp")
  done
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing $1 — on Fedora: sudo dnf install -y firewalld && sudo systemctl enable --now firewalld" >&2
    exit 1
  fi
}

need_cmd firewall-cmd

if ! systemctl is-active --quiet firewalld 2>/dev/null; then
  echo "firewalld is not running — starting it…"
  sudo systemctl enable --now firewalld
fi

zone_args=()
if [[ -n "$ZONE" ]]; then
  zone_args=(--zone="$ZONE")
else
  ZONE="$(firewall-cmd --get-default-zone)"
fi

echo "firewalld zone: $ZONE"
echo "ports: ${ports[*]}"

run_fw() {
  sudo firewall-cmd "${zone_args[@]}" "$@"
}

case "$ACTION" in
  status)
    echo "permanent:"
    run_fw --list-ports || true
    echo "runtime:"
    run_fw --list-ports || true
    for p in "${ports[@]}"; do
      if run_fw --query-port="$p" >/dev/null 2>&1; then
        echo "  OK  $p open"
      else
        echo "  —   $p closed"
      fi
    done
    ;;
  lock)
    for p in "${ports[@]}"; do
      if run_fw --permanent --query-port="$p" >/dev/null 2>&1; then
        run_fw --permanent --remove-port="$p"
        echo "removed $p"
      else
        echo "already closed $p"
      fi
    done
    run_fw --reload
    echo "locked"
    ;;
  unlock)
    for p in "${ports[@]}"; do
      if run_fw --permanent --query-port="$p" >/dev/null 2>&1; then
        echo "already open $p"
      else
        run_fw --permanent --add-port="$p"
        echo "opened $p"
      fi
    done
    run_fw --reload
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo
    echo "unlocked — from another machine:"
    echo "  test51  http://${ip:-<fedora-ip>}:${PORT_BASE}"
    echo "  tide    http://${ip:-<fedora-ip>}:${TIDE_BASE}"
    if [[ "$LAYERS" -eq 1 ]]; then
      echo "  (+ layer offsets :$((PORT_BASE+1))-:$((PORT_BASE+3)) / :$((TIDE_BASE+1))-:$((TIDE_BASE+3)))"
    fi
    ;;
esac
