#!/bin/sh
# Runs `understudy serve`, listening on 0.0.0.0:8787 unless the arguments already contain --bind.
# A first argument of -V, --version, -h, --help, or a subcommand runs the binary with the arguments unchanged
# (`serve` still gets the default --bind).
has_bind() {
  for a in "$@"; do
    case "$a" in
      --bind|--bind=*) return 0 ;;
    esac
  done
  return 1
}
case "${1:-}" in
  serve)
    shift
    if has_bind "$@"; then exec understudy serve "$@"; fi
    exec understudy serve --bind 0.0.0.0:8787 "$@" ;;
  -V|--version|-h|--help|replay|judge|train|model|status|help)
    exec understudy "$@" ;;
esac
if has_bind "$@"; then exec understudy serve "$@"; fi
exec understudy serve --bind 0.0.0.0:8787 "$@"
