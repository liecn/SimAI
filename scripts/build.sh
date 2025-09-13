#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")
ROOT_DIR=$(realpath "${SCRIPT_DIR:?}"/..)
NS3_DIR="${ROOT_DIR:?}"/ns-3-alibabacloud
SIMAI_DIR="${ROOT_DIR:?}"/astra-sim-alibabacloud
SOURCE_NS3_BIN_DIR="${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/simulation/build/scratch/ns3.36.1-AstraSimNetwork-debug
SOURCE_ANA_BIN_DIR="${SIMAI_DIR:?}"/build/simai_analytical/build/simai_analytical/SimAI_analytical
SOURCE_PHY_BIN_DIR="${SIMAI_DIR:?}"/build/simai_phy/build/simai_phynet/SimAI_phynet
SOURCE_FLOWSIM_BIN_DIR="${SIMAI_DIR:?}"/build/simai_flowsim/build/simai_flowsim/SimAI_flowsim
SOURCE_M4_BIN_DIR="${SIMAI_DIR:?}"/build/simai_m4/build/simai_m4/SimAI_m4

TARGET_BIN_DIR="${SCRIPT_DIR:?}"/../bin
function compile {
    local option="$1" 
    case "$option" in
    "ns3")
        mkdir -p "${TARGET_BIN_DIR:?}"
        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_simulator" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_simulator
        fi
        mkdir -p "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
        cp -r "${NS3_DIR:?}"/* "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3
        ./build.sh -c ns3    
        ln -s "${SOURCE_NS3_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_simulator;;
    "phy")
        mkdir -p "${TARGET_BIN_DIR:?}"
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy
        ./build.sh -c phy 
        ln -s "${SOURCE_PHY_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_phynet;;
    "analytical")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical
        ./build.sh -c analytical 
        ln -s "${SOURCE_ANA_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_analytical;;
    "flowsim")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_flowsim" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_flowsim
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr flowsim
        ./build.sh -c flowsim 
        ln -s "${SOURCE_FLOWSIM_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_flowsim;;
    "m4")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_m4" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_m4
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr m4
        ./build.sh -c m4
        ln -s "${SOURCE_M4_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_m4;;
    esac
}

function cleanup_build {
    local option="$1"
    case "$option" in
    "ns3")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_simulator" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_simulator
        fi
        rm -rf "${SIMAI_DIR:?}"/extern/network_backend/ns3-interface/
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr ns3;;
    "phy")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy;;
    "analytical")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical;;
    "flowsim")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_flowsim" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_flowsim
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr flowsim;;
    "m4")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_m4" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_m4
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -l m4;;
    esac
}

# Main Script
case "$1" in
-l|--clean)
    cleanup_build "$2";;
-c|--compile)
    compile "$2";;
-h|--help|*)
    printf -- "help message\n"
    printf -- "-c|--compile mode supported ns3/phy/analytical/flowsim/m4  (example:./build.sh -c m4)\n"
    printf -- "-l|--clean  (example:./build.sh -l flowsim)\n"
    printf -- "-lr|--clean-result mode  (example:./build.sh -lr flowsim)\n"
esac