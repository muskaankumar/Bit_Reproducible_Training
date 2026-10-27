#!/usr/bin/env bash
# Build, run the tests, and check output hashes across runs, split-K values,
# algorithms, separate processes, and every GPU on this machine.
#
# Usage: ./run_repro.sh [processes_per_gpu]
# Environment (optional):
#   SLICES="1 3"   which S values to test (default "1 3"; S=1 should reproduce
#                  the hashes of the original single-slice version)
#   GPUS="0"       which GPU indices to test (default: all)
# Output: results/<host>/run.log and results/<host>/<gpu>.hashes
# To compare machines: diff results/hostA/*.hashes results/hostB/*.hashes
set -euo pipefail
cd "$(dirname "$0")"

PROCS=${1:-3}
read -r -a SLICE_LIST <<<"${SLICES:-1 3}"
RUNS=5
SEED=12345
# Input distributions (see hash_repro_gemm.cu). Add e.g. "file:/path/A.bin,/path/B.bin"
# to test real tensors dumped from your model (shape must match the M N K below).
DISTS=(uniform normal weights wide stress)
# "M N K splits". Ragged shapes catch tile-boundary bugs; K=7 makes split_k > K clamp.
SHAPES=(
  "256 256 1024 1,2,3,4,7,8,16,64"
  "129 77 1000 1,2,3,5,7,13,64"
  "64 64 7 1,2,3,7,16"
  "1024 1024 4096 1,2,7"
)

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }

HOST=$(hostname -s)
OUT=results/$HOST
mkdir -p "$OUT"
LOG=$OUT/run.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "=== environment ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv | tee -a "$LOG"
nvcc --version | tail -n 2 | tee -a "$LOG"

# Use CMake if it is new enough for CMAKE_CUDA_ARCHITECTURES=native (>= 3.24),
# otherwise call nvcc directly (needs nvcc >= 11.6 for -arch=native).
cmake_ok() {
  command -v cmake >/dev/null || return 1
  local v; v=$(cmake --version | head -n 1 | awk '{print $3}')
  [ "$(printf '%s\n3.24\n' "$v" | sort -V | head -n 1)" = "3.24" ]
}
build_failed() { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }

log "=== build ==="
if cmake_ok; then
  cmake --version | head -n 1 | tee -a "$LOG"
  { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j; } >>"$LOG" 2>&1 || build_failed
else
  log "cmake >= 3.24 not found; building with nvcc directly"
  mkdir -p build
  NVFLAGS=(-O3 -std=c++17 -arch=native -lineinfo -I.)
  for t in test_repro_gemm bench_repro_gemm hash_repro_gemm; do
    extra=(); [ "$t" = bench_repro_gemm ] && extra=(-lcublas)
    nvcc "${NVFLAGS[@]}" repro_gemm.cu "$t.cu" -o "build/$t" ${extra[@]+"${extra[@]}"} >>"$LOG" 2>&1 || build_failed
  done
fi

# Make CUDA device numbering match nvidia-smi's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then
  read -r -a GPU_LIST <<<"$GPUS"
else
  read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
NGPU=${#GPU_LIST[@]}
FAIL=0

for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  GPU=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader | tr ' ' '_')
  HASHES=$OUT/gpu${g}_${GPU}.hashes
  : > "$HASHES"
  log "=== GPU $g: $GPU ==="

  log "--- test_repro_gemm ---"
  ./build/test_repro_gemm 2>&1 | tee -a "$LOG" || FAIL=1

  for shape in "${SHAPES[@]}"; do
    read -r M N K SPLITS <<<"$shape"
    for DIST in "${DISTS[@]}"; do
    for S in "${SLICE_LIST[@]}"; do
      declare -A seen=()
      for ((p = 0; p < PROCS; p++)); do
        log "--- hash M=$M N=$N K=$K S=$S dist=$DIST process $p ---"
        if ! ./build/hash_repro_gemm "$M" "$N" "$K" "$RUNS" "$SPLITS" "$SEED" "$DIST" ref,tiled,tc "$S" >"$OUT/tmp.txt" 2>&1; then
          FAIL=1
        fi
        cat "$OUT/tmp.txt" >>"$LOG"
        line=$(grep '^RESULT' "$OUT/tmp.txt" || echo "RESULT M=$M N=$N K=$K S=$S seed=$SEED dist=$DIST hash=none status=CRASH")
        echo "  $line" | tee -a "$LOG"
        if ((p == 0)); then grep -E '^(exact|accuracy)' "$OUT/tmp.txt" | sed 's/^/  /' | tee -a "$LOG" || true; fi
        seen["$line"]=1
      done
      if ((${#seen[@]} != 1)); then
        log "FAIL: M=$M N=$N K=$K S=$S dist=$DIST gave different hashes across processes"
        FAIL=1
      fi
      # One line per (shape, dist, S): all processes agreed, or the last one if they did not.
      echo "$line" >>"$HASHES"
      unset seen
    done
    done
  done

  # Timing. The benchmark also fails if any algorithm's bits differ from the reference.
  for S in "${SLICE_LIST[@]}"; do
  for size in "1024 1024 1024" "4096 4096 4096"; do
    log "--- bench $size S=$S ---"
    # shellcheck disable=SC2086
    ./build/bench_repro_gemm $size 1 30 tiled,tc "$S" 2>&1 | tee -a "$LOG" || FAIL=1
  done
  done
done
rm -f "$OUT/tmp.txt"

log "=== hashes per GPU ==="
for f in "$OUT"/*.hashes; do log "$f"; cat "$f" | tee -a "$LOG"; done
if ((NGPU > 1)); then
  first=$(ls "$OUT"/*.hashes | head -n 1)
  for f in "$OUT"/*.hashes; do
    cmp -s "$first" "$f" || log "NOTE: $f differs from $first (expected only if GPU models differ)"
  done
fi

if ((FAIL)); then log "OVERALL: FAIL (see $LOG)"; exit 2; fi
log "OVERALL: PASS"
