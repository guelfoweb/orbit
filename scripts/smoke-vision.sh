#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
IMAGE="${IMAGE:-/home/guelfoweb/LAB/orbit/workdir/images/vision-test-2.jpg}"

"$ORBIT" \
  --max-tokens 64 \
  --image "$IMAGE" \
  "Describe this image in one short sentence."
