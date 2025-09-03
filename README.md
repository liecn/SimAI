# 🚀 Quick Demo: NS3 vs FlowSim Performance Comparison

This repository offers two network simulation backends for SimAI with different speed vs accuracy trade-offs:

## 🔧 Build (Requires GCC-9)
```bash
# Clone and setup
git clone https://github.com/aliyun/SimAI.git && cd SimAI
git submodule update --init --recursive

# Build FlowSim backend
./scripts/build.sh -c flowsim

# Build NS3 backend  
./scripts/build.sh -c ns3
```

## ⚡ FlowSim (Fast)
```bash
time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_flowsim -w ./example/microAllReduce_16gpus.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100
```

## 🔬 NS3 (Detailed)  
```bash
time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce_16gpus.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf -r
```

## 📊 Results
- **FlowSim**: ~11s execution, 7415 cycles simulation time
- **NS3**: ~56s execution, 8072 cycles simulation time  
- **Trade-off**: FlowSim 5x faster execution, NS3 more accurate modeling