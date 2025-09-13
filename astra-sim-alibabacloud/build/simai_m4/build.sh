#!/bin/bash

# Absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BUILD_DIR="${SCRIPT_DIR:?}"/build/
RESULT_DIR="${SCRIPT_DIR:?}"/result/
BIN_DIR="${BUILD_DIR}"/SimAI_m4/
BINARY="./SimAI_m4"

# Functions
function cleanup_build {
    rm -rf "${BUILD_DIR}"
}

function cleanup_result {
    rm -rf "${RESULT_DIR}"
}

function setup {
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${RESULT_DIR}"
}

function compile {
    # Use gcc-9 like FlowSim for compatibility
    export CC=gcc-9
    export CXX=g++-9
    cd "${BUILD_DIR}" || exit
    # Configure with Release flags (no LTO to avoid Torch conflicts)
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" \
          -DUSE_ANALYTICAL=TRUE ..
    # Build in parallel using all available cores
    make -j$(nproc)
}

# Main Script
case "$1" in
-l|--clean)
    cleanup_build;;
-lr|--clean-result)
    cleanup_build
    cleanup_result;;
-c|--compile)
    setup
    compile;;
-h|--help|*)
    echo "SimAI_M4 build script."
    echo "Run $0 -c to compile.";;
esac