# 🚀 SimAI: Multi-Backend Network Simulation Framework

This repository provides three network simulation backends for SimAI, each optimized for different use cases:

## 🔧 Build (Requires GCC-9)
```bash
# Clone and setup
git clone https://github.com/aliyun/SimAI.git && cd SimAI
git submodule update --init --recursive

# Build all backends
./scripts/build.sh -c flowsim  # Fast simulation
./scripts/build.sh -c ns3      # Detailed simulation  
./scripts/build.sh -c m4       # M4 inference backend
```

## 🔧 Build using Docker
```bash
# Build a docker image
docker build -t simai .

# Clone the repo (NOT in docker)
git clone https://github.com/liecn/SimAI.git && cd SimAI
git checkout dev
git submodule update --init --recursive

# Run docker with volume binding
cd ../
docker run -it -v $(pwd)/SimAI:/data1/lichenni/projects/SimAI simai

# Build all backends (in docker)
cd /data1/lichenni/projects/SimAI/
./scripts/build.sh -c flowsim  # Fast simulation
./scripts/build.sh -c ns3      # Detailed simulation  
./scripts/build.sh -c m4       # M4 inference backend
```

## 🏃 Running Simulations

### ⚡ FlowSim (Fast Network Simulation)
```bash
time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_flowsim -w ./example/microAllReduce_16gpus.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100
```

### 🔬 NS3 (Detailed Packet-Level Simulation)
```bash
time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce_16gpus.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf -r
```

### 🧪 M4 (AI Model Inference Backend)
```bash
time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_m4 -w ./example/microAllReduce_16gpus.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100
```

### Sweep experiments
```bash
./run_sweep.sh <flowsim | ns3> <N> <M>
e.g., ./run_sweep.sh flowsim 0 1
```

## 📊 Results & Output Files

After running simulations, results are automatically saved to backend-specific directories:

### FlowSim Results: `results/flowsim/`
- `flowsim_fct.txt` - Per-flow completion times (FCT) with detailed flow information
- `EndToEnd.csv` - High-level workload completion statistics and timing breakdown

### NS3 Results: `results/ns3/`  
- `ns3_fct.txt` - Per-flow completion times with packet-level accuracy
- `EndToEnd.csv` - Workload statistics with detailed network modeling

### M4 Results: `results/m4/`
- `m4_fct.txt` - Flow completion times from M4 inference model
- `EndToEnd.csv` - Application-level performance metrics

## 🎯 Advanced Configuration

### M4 Backend Notes:
- Current implementation: Minimal stub for framework validation
- Generates fake timing data to verify AstraSim integration
- Ready for M4 inference model integration
- Use to test application ↔ network backend communication
