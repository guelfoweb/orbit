#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
MODEL="${MODEL:-gemma4:12b}"
IMAGE="${IMAGE:-/home/guelfoweb/LAB/orbit/workdir/images/vision-test-2.jpg}"

"$ORBIT" \
  --model "$MODEL" \
  --max-tokens 64 \
  --image "$IMAGE" \
  "Describe this image in one short sentence."
