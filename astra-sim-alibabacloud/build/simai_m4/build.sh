#!/usr/bin/env bash
set -euo pipefail

# Wrapper to build/clean the M4 backend from this build directory
# Usage:
#   ./build.sh -c     # configure + build
#   ./build.sh -l     # clean build outputs (keep this script)
#   ./build.sh -lr    # reserved for results cleanup (no-op)

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SRC_DIR="${SCRIPT_DIR}/../../astra-sim/network_frontend/m4"

case "${1:-}" in
  -c|--compile)
    export CC=gcc-9
    export CXX=g++-9
    mkdir -p "${SCRIPT_DIR}/build"
    cd "${SCRIPT_DIR}/build" || exit
    # Configure with Release flags for maximum runtime performance
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto -DNDEBUG" \
          -DUSE_ANALYTICAL=TRUE "${SCRIPT_DIR}"
    # Build in parallel using all available cores
    make -j$(nproc)
    # Copy binary to bin directory
    BIN_DIR="${SCRIPT_DIR}/../../bin"
    mkdir -p "${BIN_DIR}"
    if [ -f "${SCRIPT_DIR}/build/simai_m4/SimAI_m4" ]; then
      cp -f "${SCRIPT_DIR}/build/simai_m4/SimAI_m4" "${BIN_DIR}/SimAI_m4"
    fi
    ;;
  -l|--clean)
    find "${SCRIPT_DIR}" -mindepth 1 -maxdepth 1 ! -name build.sh -exec rm -rf {} +
    ;;
  -lr|--clean-result)
    ;;
  *)
    echo "Usage: $0 -c|-l|-lr"; exit 1;;
esac


