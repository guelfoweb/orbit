#!/usr/bin/env sh
set -eu

logical_cpus() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return 0
  fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

physical_cores() {
  if command -v lscpu >/dev/null 2>&1; then
    cores_per_socket="$(lscpu | awk -F: '/^Core\\(s\\) per socket:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    sockets="$(lscpu | awk -F: '/^Socket\\(s\\):/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    if [ "${cores_per_socket:-}" != "" ] && [ "${sockets:-}" != "" ]; then
      echo $((cores_per_socket * sockets))
      return 0
    fi
  fi
  if [ -r /proc/cpuinfo ]; then
    count="$(awk '
      /^physical id/ {physical=$4}
      /^core id/ {core=$4; seen[physical ":" core]=1}
      END {for (item in seen) total++; print total + 0}
    ' /proc/cpuinfo)"
    if [ "$count" -gt 0 ] 2>/dev/null; then
      echo "$count"
      return 0
    fi
  fi
  logical_cpus
}

memory_gib() {
  if [ -r /proc/meminfo ]; then
    awk '/MemTotal:/ {printf "%d\n", ($2 / 1024 / 1024) + 0.5}' /proc/meminfo
    return 0
  fi
  if command -v sysctl >/dev/null 2>&1; then
    bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    awk -v bytes="$bytes" 'BEGIN {printf "%d\n", (bytes / 1024 / 1024 / 1024) + 0.5}'
    return 0
  fi
  echo 0
}

cpu_model() {
  if [ -r /proc/cpuinfo ]; then
    awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo
    return 0
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    return 0
  fi
  echo "unknown"
}

CPUS="$(logical_cpus)"
PHYSICAL_CORES="$(physical_cores)"
RAM_GIB="$(memory_gib)"
CPU_MODEL="$(cpu_model)"

min() {
  [ "$1" -le "$2" ] && echo "$1" || echo "$2"
}

max() {
  [ "$1" -ge "$2" ] && echo "$1" || echo "$2"
}

nearest_power_profile_batch() {
  cores="$1"
  ram="$2"
  if [ "$cores" -ge 24 ] && [ "$ram" -ge 96 ]; then
    echo 512
  elif [ "$cores" -ge 8 ] && [ "$ram" -ge 32 ]; then
    echo 384
  elif [ "$cores" -ge 4 ] && [ "$ram" -ge 16 ]; then
    echo 256
  else
    echo 128
  fi
}

nearest_power_profile_ubatch() {
  cores="$1"
  ram="$2"
  if [ "$cores" -ge 16 ] && [ "$ram" -ge 64 ]; then
    echo 256
  elif [ "$cores" -ge 4 ] && [ "$ram" -ge 16 ]; then
    echo 128
  else
    echo 64
  fi
}

calculated_threads() {
  cores="$1"
  logical="$2"
  ram="$3"
  if [ "$cores" -le 2 ]; then
    echo 2
    return 0
  fi
  if [ "$ram" -lt 16 ]; then
    echo "$(min "$cores" 4)"
    return 0
  fi
  target="$cores"
  if [ "$cores" -ge 12 ]; then
    target=$((cores + cores / 3))
  fi
  target="$(min "$target" "$logical")"
  target="$(min "$target" 32)"
  target="$(max "$target" 4)"
  echo "$target"
}

calculated_cache_ram() {
  ram="$1"
  if [ "$ram" -ge 128 ]; then
    echo 32768
  elif [ "$ram" -ge 96 ]; then
    echo 24576
  elif [ "$ram" -ge 48 ]; then
    echo 8192
  elif [ "$ram" -ge 24 ]; then
    echo 6144
  elif [ "$ram" -ge 16 ]; then
    echo 4096
  else
    echo 2048
  fi
}

PROFILE="calculated conservative"
THREADS="$(calculated_threads "$PHYSICAL_CORES" "$CPUS" "$RAM_GIB")"
BATCH_SIZE="$(nearest_power_profile_batch "$PHYSICAL_CORES" "$RAM_GIB")"
UBATCH_SIZE="$(nearest_power_profile_ubatch "$PHYSICAL_CORES" "$RAM_GIB")"
CACHE_RAM="$(calculated_cache_ram "$RAM_GIB")"

# Keep measured profiles where benchmarks showed stable wins.
if [ "$PHYSICAL_CORES" -ge 24 ] && [ "$RAM_GIB" -ge 120 ]; then
  PROFILE="high-core measured"
  THREADS=32
  BATCH_SIZE=512
  UBATCH_SIZE=256
  CACHE_RAM=32768
elif [ "$PHYSICAL_CORES" -eq 6 ] && [ "$CPUS" -eq 12 ] && [ "$RAM_GIB" -ge 48 ]; then
  PROFILE="nuc10 measured"
  THREADS=6
  BATCH_SIZE=256
  UBATCH_SIZE=128
  CACHE_RAM=8192
fi

cat <<EOF
# Detected
# CPU: $CPU_MODEL
# physical_cores: $PHYSICAL_CORES
# logical_cpus: $CPUS
# memory_gib: $RAM_GIB
# profile: $PROFILE
# These values are suggestions for orbit server. Review them before exporting.
# Typical use:
#   export THREADS=$THREADS BATCH_SIZE=$BATCH_SIZE UBATCH_SIZE=$UBATCH_SIZE CACHE_RAM=$CACHE_RAM
#   orbit server --port 11976 --mtp

export THREADS=$THREADS
export BATCH_SIZE=$BATCH_SIZE
export UBATCH_SIZE=$UBATCH_SIZE
export CACHE_RAM=$CACHE_RAM
EOF
