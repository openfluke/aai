# Sourced by run-docker-lo.sh / hi / neg. Parses optional layer + cam args, sets ckpt dir.
# Usage:
#   ./run-docker-lo.sh [cam3] [layer] [--build|--logs|...]
#   ./run-docker-lo.sh --cam 3 convt2 --build
#   TEST53_CAMS=3 ./run-docker-lo.sh --build
test53_parse_layer_args() {
  TEST53_LAYER="${TEST53_LAYER:-dense}"
  TEST53_CAMS="${TEST53_CAMS:-1}"
  TEST53_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build|build|--logs|logs|--stop|stop|--status|status|--restart|restart)
        TEST53_REMAINING_ARGS+=("$1")
        shift
        ;;
      --cam)
        TEST53_CAMS="${2:?--cam needs a number}"
        shift 2
        ;;
      --cam=*)
        TEST53_CAMS="${1#*=}"
        shift
        ;;
      cam[0-9]*)
        TEST53_CAMS="${1#cam}"
        shift
        ;;
      -*)
        TEST53_REMAINING_ARGS+=("$1")
        shift
        ;;
      *)
        TEST53_LAYER="$1"
        shift
        break
        ;;
    esac
  done
  TEST53_REMAINING_ARGS+=("$@")
  if ! [[ "$TEST53_CAMS" =~ ^[0-9]+$ ]] || [[ "$TEST53_CAMS" -lt 1 ]]; then
    echo "error: TEST53_CAMS must be ≥1, got $TEST53_CAMS" >&2
    exit 1
  fi
  export TEST53_LAYER TEST53_CAMS TEST53_LAYERS="$TEST53_LAYER"
}

# Host Tide port: cam1 lo=8080 neg=8081 hi=8082; cam3 lo=8100 …
test53_tide_port() {
  local cam="${1:-1}"
  local band="${2:-lo}"
  local off=0
  case "$band" in
    neg|negative) off=1 ;;
    hi|high|extreme) off=2 ;;
  esac
  echo $((8080 + (cam - 1) * 10 + off))
}

test53_ckpt_root_for_band() {
  local band="${1:-lo}"
  local cam="${2:-1}"
  local root="test53_ckpt"
  case "$band" in
    hi|high|extreme) root="test53_ckpt_hi" ;;
    neg|negative) root="test53_ckpt_neg" ;;
  esac
  if [[ "$cam" -gt 1 ]]; then
    root="${root}_cam${cam}"
  fi
  echo "$root"
}

test53_ckpt_host_for_layer() {
  local root="${1:-test53_ckpt}"
  local dir="${2:?layer}"
  echo "${TEST53_CKPT_HOST:-$DIR/$root/$dir}"
}

test53_apply_cam_env() {
  local band="${1:-lo}"
  local cam="${TEST53_CAMS:-1}"
  local suffix=""
  [[ "$cam" -gt 1 ]] && suffix="-cam${cam}"
  export TEST53_CAMS="$cam"
  export TIDE_PORT="${TIDE_PORT:-$(test53_tide_port "$cam" "$band")}"
  export TEST53_CONTAINER_NAME="${TEST53_CONTAINER_NAME:-test53${suffix}-${band}}"
  export TEST53_PROJECT="${TEST53_PROJECT:-test53${suffix}-${band}}"
}
