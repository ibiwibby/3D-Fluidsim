#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this script inside an allocated GPU session (idev)." >&2
    exit 1
fi
choice="${1:-both}"
case "$choice" in periodic|closed|both) ;; *)
    echo "Usage: bash tacc/run64.sh [periodic|closed|both]" >&2
    exit 1 ;;
esac
arch="${CUDA_ARCH:-sm_80}"
steps="${CHANNEL64_STEPS:-600}"
output_every="${CHANNEL64_OUTPUT_EVERY:-5}"
command -v nvcc >/dev/null || { echo "Load the site's CUDA module first." >&2; exit 1; }
make CUDA_ARCH="$arch" DEBUG=0 all
binary="./build/$arch/debug-0/fluid3d"
"$binary" --device-info
"$binary" --check
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for boundary in periodic closed; do
    if [[ "$choice" != both && "$choice" != "$boundary" ]]; then continue; fi
    options=()
    if [[ "$boundary" == closed ]]; then options+=(--closed-ends); fi
    output="../results/channel64_${boundary}_${SLURM_JOB_ID}_$stamp"
    mkdir -p "$(dirname "$output")"
    "$binary" --channel --size 64 --radius 8 --force 1.25e-6 \
        --steps "$steps" --output-every "$output_every" "${options[@]}" --output "$output" \
        2>&1 | tee "${output}.log"
    echo "Completed output: $(realpath "$output")"
done
