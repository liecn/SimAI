# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")
NS3_BUILD_DIR="${SCRIPT_DIR:?}"/build/astra_ns3
SIMAI_PHY_BUILD_DIR="${SCRIPT_DIR:?}"/build/simai_phy
SIMAI_ANALYTICAL_BUILD_DIR="${SCRIPT_DIR:?}"/build/simai_analytical
SIMAI_FLOWSIM_BUILD_DIR="${SCRIPT_DIR:?}"/build/simai_flowsim
SIM_LOG_DIR="${SCRIPT_DIR:?}"/../logs

# Functions
function cleanup_build {
    local option="$1"
    case "$option" in
    "ns3")
        cd "${NS3_BUILD_DIR}"
        ./build.sh -l;;
    "phy")
        cd "${SIMAI_PHY_BUILD_DIR}"
        ./build.sh -l;;
    "analytical")
        cd "${SIMAI_ANALYTICAL_BUILD_DIR}"
        ./build.sh -l;;
    "flowsim")
        cd "${SIMAI_FLOWSIM_BUILD_DIR}"
        ./build.sh -l;;
    esac
}

function cleanup_result {
    local option="$1"
    case "$option" in
    "ns3")
        cd "${NS3_BUILD_DIR}"
        ./build.sh -lr;;
    "phy")
        cd "${SIMAI_PHY_BUILD_DIR}"
        ./build.sh -lr;;
    "analytical")
        cd "${SIMAI_ANALYTICAL_BUILD_DIR}"
        ./build.sh -lr;;
    "flowsim")
        cd "${SIMAI_FLOWSIM_BUILD_DIR}"
        ./build.sh -lr;;
    esac
}

function compile {
    mkdir -p "${SIM_LOG_DIR}"/inputs/system/
    mkdir -p "${SIM_LOG_DIR}"/inputs/workload/
    mkdir -p "${SIM_LOG_DIR}"/simulation/
    mkdir -p "${SIM_LOG_DIR}"/config/
    mkdir -p "${SIM_LOG_DIR}"/topo/
    mkdir -p "${SIM_LOG_DIR}"/results/
    local option="$1" 
    case "$option" in
    "ns3")
        cd "${NS3_BUILD_DIR}"
        ./build.sh -c;;
    "phy")
        cd "${SIMAI_PHY_BUILD_DIR}"
        ./build.sh -c RDMA;;
    "analytical")
        cd "${SIMAI_ANALYTICAL_BUILD_DIR}"
        ./build.sh -c;;
    "flowsim")
        cd "${SIMAI_FLOWSIM_BUILD_DIR}"
        ./build.sh -c;;
    esac
}

# Main Script
case "$1" in
-l|--clean)
    cleanup_build "$2";;
-lr|--clean-result)
    cleanup_result "$2";;
-c|--compile)
    compile "$2";;
-h|--help|*)
    printf -- "help message\n"
    printf -- "-c|--compile mode supported ns3/phy/analytical/flowsim  (example:./build.sh -c flowsim)\n"
    printf -- "-l|--clean  (example:./build.sh -l flowsim)\n"
    printf -- "-lr|--clean-result mode  (example:./build.sh -lr flowsim)\n"
esac