#!/usr/bin/env bash
# Start one test51 process per selected layer.
# Each layer gets its own ports + checkpoint folder.
#
# Usage:
#   ./start.sh                    # layers from TEST51_LAYERS in .env (or menu)
#   ./start.sh dense
#   ./start.sh dense,dense-wide
#   ./start.sh all
#   ./start.sh -i                 # interactive picker
#   ./start.sh dense -- -autostart  # extra args after -- go to the binary
#
# Layout:
#   run/<layer>.pid  run/<layer>.log
#   test51_ckpt/<layer>/{progress,history,results}.json
#   ports: TEST51_PORT_BASE+i  /  TIDE_PORT_BASE+i
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

ALL_LAYERS=(dense dense-wide dense-deep dense-deep-wide)

if [[ -f "$DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DIR/.env"
  set +a
fi

CKPT_ROOT="${TEST51_CKPT_ROOT:-test51_ckpt}"
PORT_BASE="${TEST51_PORT_BASE:-5151}"
TIDE_BASE="${TIDE_PORT_BASE:-8080}"
RUN_DIR="${TEST51_RUN_DIR:-run}"
BIN="$DIR/test51"
INTERACTIVE=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: ./start.sh [layers|all|-i] [-- binary-args...]

Layers: dense | dense-wide | dense-deep | dense-deep-wide | all
  (comma-separated OK)

Env (.env):
  TEST51_LAYERS=dense,dense-wide   # default when no args
  TEST51_MODES=NormalBP
  TEST51_PORT_BASE=5151            # +0,+1,… per selected layer
  TIDE_PORT_BASE=8080
  TEST51_CKPT_ROOT=test51_ckpt     # → <root>/<layer>/
  TEST51_AUTOSTART=false
EOF
}

# Parse args: layers / -i / -- extras
LAYER_SPEC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -i|--interactive) INTERACTIVE=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -*)
      # bare binary flags without -- (compat)
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      if [[ -z "$LAYER_SPEC" ]]; then
        LAYER_SPEC="$1"
      else
        LAYER_SPEC="$LAYER_SPEC,$1"
      fi
      shift
      ;;
  esac
done

pick_interactive() {
  echo "Select layers (space-separated numbers, or 'all'):"
  local i=1
  for L in "${ALL_LAYERS[@]}"; do
    printf "  %d) %s\n" "$i" "$L"
    i=$((i + 1))
  done
  echo -n "> "
  read -r ans
  if [[ -z "$ans" ]]; then
    echo "nothing selected"
    exit 1
  fi
  if [[ "$ans" == "all" ]]; then
    LAYER_SPEC="all"
    return
  fi
  local out=()
  for n in $ans; do
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n >= 1 && n <= ${#ALL_LAYERS[@]} )); then
      out+=("${ALL_LAYERS[$((n - 1))]}")
    else
      echo "bad pick: $n" >&2
      exit 1
    fi
  done
  LAYER_SPEC=$(IFS=,; echo "${out[*]}")
}

if [[ "$INTERACTIVE" -eq 1 ]]; then
  pick_interactive
fi

if [[ -z "$LAYER_SPEC" ]]; then
  LAYER_SPEC="${TEST51_LAYERS:-}"
fi
if [[ -z "$LAYER_SPEC" ]]; then
  pick_interactive
fi

# Expand to array
LAYERS=()
if [[ "${LAYER_SPEC,,}" == "all" ]]; then
  LAYERS=("${ALL_LAYERS[@]}")
else
  IFS=',' read -ra RAW <<<"$LAYER_SPEC"
  for tok in "${RAW[@]}"; do
    tok="$(echo "$tok" | tr '[:upper:]' '[:lower:]' | xargs)"
    [[ -z "$tok" ]] && continue
    ok=0
    for L in "${ALL_LAYERS[@]}"; do
      if [[ "$tok" == "$L" ]]; then ok=1; break; fi
    done
    if [[ "$ok" -ne 1 ]]; then
      echo "unknown layer: $tok (want: ${ALL_LAYERS[*]})" >&2
      exit 2
    fi
    # dedupe
    skip=0
    for e in "${LAYERS[@]+"${LAYERS[@]}"}"; do
      [[ "$e" == "$tok" ]] && skip=1 && break
    done
    [[ "$skip" -eq 1 ]] && continue
    LAYERS+=("$tok")
  done
fi

if [[ ${#LAYERS[@]} -eq 0 ]]; then
  echo "no layers selected" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$CKPT_ROOT"

if [[ ! -x "$BIN" ]]; then
  echo "building binary..."
  "$DIR/build.sh"
fi

ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "════════════════════════════════════════════════════"
echo " test51 · layers: ${LAYERS[*]}"
echo " modes:  ${TEST51_MODES:-all}"
echo " ckpt:   $CKPT_ROOT/<layer>/"
echo "════════════════════════════════════════════════════"

# Stable port index per layer (dense=0 …) so status/start always match.
layer_index() {
  local want="$1" i=0
  for L in "${ALL_LAYERS[@]}"; do
    if [[ "$L" == "$want" ]]; then
      echo "$i"
      return
    fi
    i=$((i + 1))
  done
  echo "-1"
}

idx=0
for layer in "${LAYERS[@]}"; do
  pidfile="$RUN_DIR/$layer.pid"
  logfile="$RUN_DIR/$layer.log"
  ckpt="$CKPT_ROOT/$layer"
  idx="$(layer_index "$layer")"
  if [[ "$idx" -lt 0 ]]; then
    echo "internal: bad layer $layer" >&2
    exit 1
  fi
  p51=$((PORT_BASE + idx))
  ptide=$((TIDE_BASE + idx))
  mkdir -p "$ckpt"

  if [[ -f "$pidfile" ]]; then
    old="$(cat "$pidfile")"
    if kill -0 "$old" 2>/dev/null; then
      echo "· $layer already running (pid $old)  :$p51 / :$ptide"
      continue
    fi
    rm -f "$pidfile"
  fi

  echo "· starting $layer"
  echo "    test51  http://${ip:-127.0.0.1}:$p51"
  if [[ -n "${TIDE_PORT_BASE:-8080}" && "${TIDE_PORT_BASE}" != "off" && "${TIDE_PORT_BASE}" != "0" ]]; then
    echo "    tide    http://${ip:-127.0.0.1}:$ptide"
  fi
  echo "    ckpt    $ckpt/"
  echo "    log     $logfile"

  # Layer-specific overrides (export for this child only).
  tide_addr="0.0.0.0:$ptide"
  if [[ -z "${TIDE_PORT_BASE:-8080}" || "${TIDE_PORT_BASE}" == "off" || "${TIDE_PORT_BASE}" == "0" ]]; then
    tide_addr=""
  fi

  env \
    TEST51_LAYERS="$layer" \
    TEST51_ADDR="0.0.0.0:$p51" \
    TIDE_ADDR="$tide_addr" \
    TEST51_CKPT="$ckpt" \
    TEST51_FULL="${TEST51_FULL:-true}" \
    nohup "$BIN" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" >>"$logfile" 2>&1 &
  echo $! >"$pidfile"

  sleep 0.4
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  FAILED — tail $logfile:" >&2
    tail -30 "$logfile" 2>/dev/null || true
    rm -f "$pidfile"
    exit 1
  fi
done

# Also write a manifest of what we launched
{
  echo "# test51 layers $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "layers=${LAYERS[*]}"
  for layer in "${LAYERS[@]}"; do
    idx="$(layer_index "$layer")"
    echo "$layer pid=$(cat "$RUN_DIR/$layer.pid" 2>/dev/null || echo ?) port=$((PORT_BASE + idx)) tide=$((TIDE_BASE + idx)) ckpt=$CKPT_ROOT/$layer"
  done
} >"$RUN_DIR/manifest.txt"

echo
echo "stop:   ./stop.sh            # all"
echo "        ./stop.sh dense      # one layer"
echo "status: ./status.sh"
echo "logs:   tail -f $RUN_DIR/<layer>.log"
