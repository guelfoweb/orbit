#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
AUDIO="${AUDIO:-/home/guelfoweb/LAB/orbit/workdir/audio/voice-sample-16k-mono.wav}"

"$ORBIT" \
  --max-tokens 128 \
  --audio "$AUDIO" \
  "Transcribe this audio. Return only the transcript."
