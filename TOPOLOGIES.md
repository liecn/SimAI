# Commercial Network Topologies for Evaluation

This document describes the commercial datacenter network topologies available for co-optimization experiments.

---

## Available Topologies

### 1. **Custom Multi-Rail Topology** (baseline)
- **File**: `example/topo.txt`
- **Scale**: 32 GPUs (4 servers × 8 GPUs/server)
- **Architecture**: Multi-rail with explicit rail switches
- **Hierarchy**: GPUs → NVSwitches (L1) → Rail switches (L2) → Spine
- **Bandwidth Tiers**:
  - GPU ↔ NVSwitch: 2400 Gbps (NVLink)
  - Rail ↔ Spine: 400 Gbps (network backbone)
  - GPU ↔ Rail: 100 Gbps (NIC, 4 rails per server)
- **Total nodes**: 45 (32 GPUs + 4 NVSwitches + 8 rail switches + 1 spine)
- **Use case**: Preliminary experiments, algorithm development

---

### 2. **NVIDIA Spectrum-X** (rail-optimized)
- **Files**: 
  - `example/Spectrum-X_32g_8gps_100Gbps_A100` (32 GPUs)
  - `example/Spectrum-X_64g_8gps_100Gbps_A100` (64 GPUs)
  - `example/Spectrum-X_128g_8gps_100Gbps_A100` (128 GPUs)
- **Architecture**: Rail-optimized for modern AI workloads
- **Characteristics**:
  - **Rail-optimized**: Multi-rail NICs per GPU for higher bisection bandwidth
  - **NVLink**: 2880 Gbps (higher than baseline)
  - **NIC bandwidth**: 100 Gbps per rail
  - **Larger switch fabric**: More switches for better scalability
- **Topology details (32 GPU)**:
  - Total nodes: 108 (32 GPUs + 4 NVSwitches + 72 network switches)
  - Total links: 576
  - Servers: 4 (8 GPUs/server)
- **Use case**: 
  - Baseline comparison (32 GPU vs. custom topology)
  - Scalability experiments (64, 128 GPUs)
  - Representative of NVIDIA AI cluster designs

---

### 3. **DCN+ (Traditional DCN)** (non-rail-optimized)
- **File**: `example/DCN+SingleToR_64g_8gps_100Gbps_A100`
- **Scale**: 64 GPUs (8 servers × 8 GPUs/server)
- **Architecture**: Traditional data center network (non-rail-optimized)
- **Characteristics**:
  - **Single-rail**: One NIC per GPU (simpler than Spectrum-X)
  - **Simpler switching fabric**: Fewer switches (9 vs 72 for Spectrum-X)
  - **Traditional ToR design**: More similar to classic HPC networks
- **Topology details**:
  - Total nodes: 81 (64 GPUs + 8 NVSwitches + 9 network switches)
  - Total links: 136 (much fewer than Spectrum-X)
  - Servers: 8 (8 GPUs/server)
- **Use case**: 
  - Architectural diversity in evaluation
  - Show co-optimization works on non-rail-optimized networks
  - Representative of traditional HPC/cloud cluster designs

---

## Key Differences Summary

| Topology | GPUs | Architecture | NVLink BW | NIC BW | Network Switches | Total Links | Use Case |
|----------|------|--------------|-----------|--------|------------------|-------------|----------|
| Custom | 32 | Multi-rail | 2400 Gbps | 100 Gbps | 9 (8 rail + 1 spine) | 72 | Development |
| Spectrum-X 32 | 32 | Rail-optimized | 2880 Gbps | 100 Gbps | 72 | 576 | Baseline |
| Spectrum-X 64 | 64 | Rail-optimized | 2880 Gbps | 100 Gbps | 72 | ~1100 | Scalability |
| Spectrum-X 128 | 128 | Rail-optimized | 2880 Gbps | 100 Gbps | 72 | ~2200 | Scalability |
| DCN+ 64 | 64 | Single-rail | 2880 Gbps | 100 Gbps | 9 | 136 | Diversity |

---

## Recommended Evaluation Plan

### Phase 1: Algorithm Development (Custom 32-GPU)
- **Topology**: `topo.txt` (custom 32-GPU)
- **Goal**: Develop and debug co-optimization algorithms
- **Workloads**: All 5 (GPT-3 Small/Medium/Large, Multi-Tenant, Bursty)
- **Duration**: Fast iteration cycles

### Phase 2: Baseline Comparison (Spectrum-X 32-GPU)
- **Topology**: `Spectrum-X_32g_8gps_100Gbps_A100`
- **Goal**: Compare against commercial topology at same scale
- **Workloads**: All 5
- **Analysis**: Does co-optimization generalize to commercial topologies?

### Phase 3: Scalability Study (Spectrum-X 64/128-GPU)
- **Topologies**: `Spectrum-X_64g_8gps_100Gbps_A100`, `Spectrum-X_128g_8gps_100Gbps_A100`
- **Goal**: Show co-optimization scales to larger clusters
- **Workloads**: GPT-3 Medium/Large, Multi-Tenant (adjust DP/TP accordingly)
- **Analysis**: Does coordination overhead increase with scale?

### Phase 4: Architectural Robustness (DCN+ 64-GPU)
- **Topology**: `DCN+SingleToR_64g_8gps_100Gbps_A100`
- **Goal**: Show co-optimization works on different network architectures
- **Workloads**: GPT-3 Medium/Large
- **Analysis**: Do learned templates transfer across architectures?

---

## Usage Examples

### Run simulation on Spectrum-X 32-GPU:
```bash
python run_simai.py ns3 \
  --topo example/Spectrum-X_32g_8gps_100Gbps_A100 \
  --workload example/microAllReduce.txt \
  --results-dir results/spectrum_x_32
```

### Run simulation on DCN+ 64-GPU:
```bash
python run_simai.py ns3 \
  --topo example/DCN+SingleToR_64g_8gps_100Gbps_A100 \
  --workload example/microAllReduce.txt \
  --results-dir results/dcn_plus_64
```

### Visualize commercial topology:
```bash
python topo_viz.py \
  --topo example/Spectrum-X_32g_8gps_100Gbps_A100 \
  --out spectrum_x_32_topology.png
```

---

## For MLSys 2026 Paper

**Suggested presentation**:

1. **Main results**: Use Spectrum-X 32-GPU topology for all baseline comparisons
   - Most representative of modern AI clusters
   - Commercial validation (not just custom topology)

2. **Scalability**: Show 32 → 64 → 128 GPU results
   - Demonstrate coordination doesn't break at scale
   - Plot: Improvement % vs. scale

3. **Generalization**: Compare Spectrum-X vs. DCN+ at 64 GPUs
   - Show framework works across architectures
   - Emphasize template learning transfers

4. **Ablation**: Use Spectrum-X 32-GPU
   - Isolate contribution of each component

This gives you a **strong empirical evaluation** with commercial topologies at multiple scales!

