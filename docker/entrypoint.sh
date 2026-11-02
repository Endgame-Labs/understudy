#!/bin/sh
# Runs `understudy serve`, listening on 0.0.0.0:8787 unless the arguments already contain --bind.
for a in "$@"; do
  case "$a" in
    --bind|--bind=*) exec understudy serve "$@" ;;
  esac
done
exec understudy serve --bind 0.0.0.0:8787 "$@"
