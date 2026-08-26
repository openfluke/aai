# Sourced by run-docker-lo.sh / hi / neg. Parses optional layer arg, sets ckpt dir.
# Usage: ./run-docker-lo.sh [layer] [--build|--logs|...]
#   ./run-docker-lo.sh convt2 --build
test53_parse_layer_args() {
  TEST53_LAYER="${TEST53_LAYER:-dense}"
  TEST53_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build|build|--logs|logs|--stop|stop|--status|status|--restart|restart)
        TEST53_REMAINING_ARGS+=("$1")
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
  export TEST53_LAYER
  export TEST53_LAYERS="$TEST53_LAYER"
}

test53_ckpt_host_for_layer() {
  local root="${1:-test53_ckpt}"
  local dir="${2:?layer}"
  echo "${TEST53_CKPT_HOST:-$DIR/$root/$dir}"
}
