# SimAI: Multi-Backend Network Simulation for Distributed ML

This repository provides scripts and tools to run distributed ML simulations using three different network simulation backends: **UNISON (ns-3)**, **flowSim**, and **m4**.

---

## Repository Structure

```
SimAI/
├── astra-sim-alibabacloud/      # Core simulation framework
│   ├── astra-sim/               # AstraSim system layer
│   │   ├── network_frontend/    # Network backend implementations
│   │   │   ├── ns3/             # UNISON (ns-3) packet-level simulator
│   │   │   ├── flowsim/         # flowSim analytical simulator
│   │   │   └── m4/              # m4 ML-based simulator
│   │   └── system/              # System components (routing, collectives, GPU grouping)
│   ├── extern/                  # ns-3 source code
│   └── build.sh                 # Build script for all backends
├── example/
│   ├── topo.txt                 # Network topology (32 GPUs, 4 rails, multi-tenant)
│   ├── microAllReduce.txt       # AllReduce collective workload
│   └── SimAI.conf               # ns-3 configuration
├── results/                     # Simulation outputs (organized by backend)
│   ├── flowsim/                 # flowSim results
│   ├── ns3/                     # ns-3 results (includes fct.txt with per-flow stats)
│   └── m4/                      # m4 results
├── scripts/                     # Build and utility scripts
├── run_simai.py                 # Main simulation runner
└── topo_viz.py                  # Topology visualization tool
```

---

## Quick Start

### 1. Setup Environment

**Activate Python environment:**
```bash
cd /path/to/ADRS
source .venv/bin/activate
```

**Install GCC-9 (required for compilation):**
```bash
sudo apt-get install gcc-9 g++-9
```

### 2. Build Simulation Backends

Navigate to the SimAI directory and build the backends:

```bash
cd openevolve/examples/m5/SimAI

# Build each backend (choose one or all)
./scripts/build.sh -c flowsim   # Fast analytical simulator
./scripts/build.sh -c ns3       # Packet-level simulator (ground truth)
./scripts/build.sh -c m4        # ML-based simulator (requires CUDA)
```

**Build outputs:**
- `bin/SimAI_flowsim` — flowSim executable
- `bin/SimAI_simulator` — ns-3 executable
- `bin/SimAI_m4` — m4 executable

### 3. Run Simulations

Use `run_simai.py` to run simulations with the default topology (`example/topo.txt`):

```bash
# Run flowSim (fastest, ~1 second)
python run_simai.py flowsim

# Run ns-3 (packet-level, ~1-2 minutes)
python run_simai.py ns3

# Run m4 (ML-based, requires GPU, ~10-30 seconds)
python run_simai.py m4 --gpu 0
```

**Custom options:**
```bash
# Use custom topology, workload, or results directory
python run_simai.py ns3 \
  --topo example/topo.txt \
  --workload example/microAllReduce.txt \
  --results-dir results

# Run m4 on specific GPU
python run_simai.py m4 --gpu 1
```

### 4. Check Results

Results are organized by backend in the `results/` directory:

```bash
results/
├── flowsim/
│   ├── EndToEnd.csv         # Completion times per GPU
│   ├── run.log              # Execution log
│   └── runtime.txt          # Simulation metadata
├── ns3/
│   ├── EndToEnd.csv         # Completion times per GPU
│   ├── fct.txt              # Per-flow statistics (src, dst, size, FCT, route)
│   ├── run.log
│   └── runtime.txt
└── m4/
    ├── EndToEnd.csv
    ├── run.log
    └── runtime.txt
```

**Key output files:**
- `EndToEnd.csv` — Per-GPU completion times for the AllReduce collective
- `fct.txt` (ns-3 only) — Per-flow network statistics including routes taken
- `runtime.txt` — Simulation duration and status

---

## Topology Visualization

Visualize the network topology to understand the hierarchical structure:

```bash
python topo_viz.py --topo example/topo.txt --out topology.png
```

**The 32-GPU topology features:**
- **32 GPUs** organized in 2 groups (Group A: 0-15, Group B: 16-31)
- **4 NVSwitches** (L1) for intra-node GPU-to-GPU communication (2400 Gbps NVLink)
- **8 Rail switches** (L2) for multi-rail inter-node networking (100 Gbps per GPU-rail link)
- **1 Spine switch** connecting all rail switches (400 Gbps per rail-spine link)

This multi-rail topology creates realistic network contention when running distributed ML workloads.

---

## Network Topology Details

The `topo.txt` file defines a **32-GPU, 4-rail datacenter network** with three bandwidth tiers:

1. **2400 Gbps (NVLink)** — GPU ↔ NVSwitch intra-node links
2. **400 Gbps** — Rail ↔ Spine network backbone
3. **100 Gbps (NIC)** — GPU ↔ Rail inter-node links (multi-rail)

**Key features for research:**
- **Multi-tenant simulation**: With DP group size = 16, the 32 GPUs create 2 concurrent AllReduce rings that compete for network resources
- **Realistic contention**: Multiple concurrent collective operations create realistic network congestion
- **Per-flow tracking**: ns-3 backend exports detailed flow statistics including routes, FCT, and ideal FCT

---

## Co-Optimization Research

This SimAI setup enables research on **co-optimizing multiple interacting algorithms** for distributed ML:

### Supported Optimization Dimensions

1. **Collective Communication Scheduling** (`system/MockNcclGroup.cc`)
   - Optimize AllReduce/AllGather/ReduceScatter scheduling
   - Current: Ring-based, pipelined chunk scheduling

2. **Network Routing** (`system/routing/src/RoutingFramework.cc`)
   - Optimize path selection for network flows
   - Current: ECMP (Equal-Cost Multi-Path)
   - Custom paths injectable via `flow_to_path_map_`

3. **GPU Grouping Strategy** (`system/MockNcclGroup.cc`, line 78)
   - Optimize how GPUs are assigned to DP/TP groups
   - Current: Interleaved grouping (DP ranks: 0,1,2,...,15; 16,17,...,31)

### End-to-End Evaluation

All three algorithms interact and affect the end-to-end training performance. The simulation framework allows:
- **Global objective optimization**: Minimize total AllReduce completion time
- **Cross-algorithm coordination**: Changes in routing affect collective scheduling
- **Realistic contention modeling**: Multi-tenant workloads capture real network dynamics

---

## Advanced Usage

### Modify Network Topology

Edit `example/topo.txt` to change:
- Number of GPUs
- Number of rail switches
- Link bandwidths
- Link latencies

Format: See existing `topo.txt` for structure.

### Modify Workload

Edit `example/microAllReduce.txt` to change:
- Message sizes
- Collective operation types
- Number of iterations

### Custom Routing

Inject custom paths by modifying `RoutingFramework.cc`:
```cpp
// In GetFlowSimPathByNodeIds()
if (flow_to_path_map_.find(key) != flow_to_path_map_.end()) {
    return flow_to_path_map_[key];  // Use custom path
}
```

### Custom GPU Grouping

Modify `MockNcclGroup.cc` line 78 to change GPU assignment:
```cpp
// Current (interleaved):
int rank = i + j*DP_nums;

// Alternative (contiguous):
int rank = j + i*TP_nums;
```