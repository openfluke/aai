# Sourced by run-docker-lo.sh / hi / neg. Parses optional layer + cam args, sets ckpt dir.
# Default: both cam1 + cam3 (one machine runs lo or hi for both cams).
# Usage:
#   ./run-docker-lo.sh mamba --build              # cam1 + cam3
#   ./run-docker-lo.sh cam3 mamba --build         # cam3 only
#   ./run-docker-hi.sh mamba --build              # cam1-hi + cam3-hi
#   TEST54_CAMS=3 ./run-docker-lo.sh --build
test54_parse_layer_args() {
  TEST54_LAYER="${TEST54_LAYER:-mamba}"
  TEST54_CAMS="${TEST54_CAMS:-both}"
  TEST54_CAMS_LIST=()
  TEST54_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build|build|--logs|logs|--stop|stop|--status|status|--restart|restart)
        TEST54_REMAINING_ARGS+=("$1")
        shift
        ;;
      --cam)
        TEST54_CAMS="${2:?--cam needs a number}"
        TEST54_CAMS_LIST=("$TEST54_CAMS")
        shift 2
        ;;
      --cam=*)
        TEST54_CAMS="${1#*=}"
        TEST54_CAMS_LIST=("$TEST54_CAMS")
        shift
        ;;
      both|cam1+3|cams)
        TEST54_CAMS_LIST=(1 3)
        TEST54_CAMS=1
        shift
        ;;
      cam1,3|cam1+cam3)
        TEST54_CAMS_LIST=(1 3)
        TEST54_CAMS=1
        shift
        ;;
      cam[0-9]*)
        TEST54_CAMS="${1#cam}"
        TEST54_CAMS_LIST=("$TEST54_CAMS")
        shift
        ;;
      -*)
        TEST54_REMAINING_ARGS+=("$1")
        shift
        ;;
      *)
        TEST54_LAYER="$1"
        shift
        break
        ;;
    esac
  done
  TEST54_REMAINING_ARGS+=("$@")
  if [[ ${#TEST54_CAMS_LIST[@]} -eq 0 ]]; then
    case "$TEST54_CAMS" in
      both|cam1+3|cams|all|cam1,3|cam1+cam3)
        TEST54_CAMS_LIST=(1 3)
        ;;
      *)
        TEST54_CAMS_LIST=("$TEST54_CAMS")
        ;;
    esac
  fi
  for c in "${TEST54_CAMS_LIST[@]}"; do
    if ! [[ "$c" =~ ^[0-9]+$ ]] || [[ "$c" -lt 1 ]]; then
      echo "error: TEST54_CAMS must be ≥1, got $c" >&2
      exit 1
    fi
  done
  TEST54_CAMS="${TEST54_CAMS_LIST[0]}"
  export TEST54_LAYER TEST54_CAMS TEST54_LAYERS="$TEST54_LAYER"
  export TEST54_CAMS_LIST
}

# Host Tide port (offset from test53):
#   cam1 lo=9080 neg=9081 hi=9082
#   cam3 lo=9100 neg=9101 hi=9102
test54_tide_port() {
  local cam="${1:-1}"
  local band="${2:-lo}"
  local off=0
  case "$band" in
    neg|negative) off=1 ;;
    hi|high|extreme) off=2 ;;
  esac
  echo $((9080 + (cam - 1) * 10 + off))
}

test54_ckpt_root_for_band() {
  local band="${1:-lo}"
  local cam="${2:-1}"
  local root="test54_ckpt"
  case "$band" in
    hi|high|extreme) root="test54_ckpt_hi" ;;
    neg|negative) root="test54_ckpt_neg" ;;
  esac
  if [[ "$cam" -gt 1 ]]; then
    root="${root}_cam${cam}"
  fi
  echo "$root"
}

test54_ckpt_host_for_layer() {
  local root="${1:-test54_ckpt}"
  local dir="${2:?layer}"
  echo "${TEST54_CKPT_HOST:-$DIR/$root/$dir}"
}

test54_apply_cam_env() {
  local band="${1:-lo}"
  local cam="${TEST54_CAMS:-1}"
  local suffix=""
  [[ "$cam" -gt 1 ]] && suffix="-cam${cam}"
  export TEST54_CAMS="$cam"
  export TIDE_PORT="${TIDE_PORT:-$(test54_tide_port "$cam" "$band")}"
  export TEST54_CONTAINER_NAME="${TEST54_CONTAINER_NAME:-test54${suffix}-${band}}"
  export TEST54_PROJECT="${TEST54_PROJECT:-test54${suffix}-${band}}"
}

# Run the same band for every cam in TEST54_CAMS_LIST (cam1 + cam3 when both).
# Clears TIDE_PORT / container / project between cams so they don't collide.
test54_run_for_cams() {
  local band="${1:?band}"
  local override="${2:?compose override}"
  shift 2
  local cam root
  for cam in "${TEST54_CAMS_LIST[@]}"; do
    unset TIDE_PORT TEST54_CONTAINER_NAME TEST54_PROJECT TEST54_CKPT_HOST || true
    export TEST54_CAMS="$cam"
    test54_apply_cam_env "$band"
    root="$(test54_ckpt_root_for_band "$band" "$cam")"
    export TEST54_CKPT_HOST="$(test54_ckpt_host_for_layer "$root" "$TEST54_LAYER")"
    export TEST54_COMPOSE_OVERRIDE="$override"
    mkdir -p "$TEST54_CKPT_HOST"
    echo "test54 ${band} · cam=$cam · layer=$TEST54_LAYER  depth=${TEST54_DEPTH:-4}  lrs=${TEST54_LRS:-funny-lo}  dur=${TEST54_DURATION:-15s}  ckpt=$TEST54_CKPT_HOST  tide=:$TIDE_PORT"
    "$DIR/run-docker.sh" "$@"
  done
}
