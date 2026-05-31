#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building SM80 Fallback Unit Tests ==="
echo "Project dir: $PROJECT_DIR"

nvcc \
    -DDISABLE_NVSHMEM \
    -I"${PROJECT_DIR}/csrc" \
    -gencode=arch=compute_80,code=sm_80 \
    -std=c++17 \
    -O3 \
    -o "${SCRIPT_DIR}/test_sm80_fallback" \
    "${SCRIPT_DIR}/test_sm80_fallback.cu"

echo "Build succeeded."

echo ""
echo "=== Running SM80 Fallback Unit Tests ==="
"${SCRIPT_DIR}/test_sm80_fallback"