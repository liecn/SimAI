#!/bin/bash
# Topology Generation Script for Co-Optimization Research
# Generates 5 topologies for testing multi-algorithm co-optimization effectiveness

set -e

echo "========================================="
echo "Generating Co-Optimization Topologies"
echo "========================================="
echo ""

# Change to topology generator directory
cd "$(dirname "$0")/astra-sim-alibabacloud/inputs/topo"

# Output directly to SimAI example/
OUTPUT_DIR="../../../example"

# =============================================================================
# Topology 1: DCN+ Single-ToR 64 GPU - ARCHITECTURE CONTRAST
# Role: Test on non-rail-optimized architecture (traditional HPC)
# Key: Single ToR → uplink bottleneck, placement+scheduling matter MORE
# =============================================================================
echo "[1/5] Generating T1: DCN+ Single-ToR 64 GPU..."
python gen_Topo_Template.py \
  --topology DCN+ \
  --gpu 64 \
  --gpu_per_server 8 \
  --gpu_type A100 \
  --bandwidth 100Gbps \
  --ap_bandwidth 400Gbps \
  --nvlink_bw 2880Gbps

if [ -f "DCN+SingleToR_64g_8gps_100Gbps_A100" ]; then
  cp "DCN+SingleToR_64g_8gps_100Gbps_A100" "$OUTPUT_DIR/T1_DCN+SingleToR_64g_8gps_100Gbps_A100"
  rm "DCN+SingleToR_64g_8gps_100Gbps_A100"
  echo "✓ Saved to $OUTPUT_DIR/T1_DCN+SingleToR_64g_8gps_100Gbps_A100"
else
  echo "✗ Failed to generate DCN+ 64 GPU"
  exit 1
fi
echo ""

# =============================================================================
# Topology 2: AlibabaHPN Dual-ToR 64 GPU - PRODUCTION CLOUD
# Role: Test on production-like multi-tenant architecture
# Key: Dual-ToR → asymmetric paths, latency variance, routing critical
# =============================================================================
echo "[2/5] Generating T2: AlibabaHPN Dual-ToR 64 GPU..."
python gen_Topo_Template.py \
  --topology AlibabaHPN \
  --dt \
  --gpu 64 \
  --gpu_per_server 8 \
  --gpu_type A100 \
  --bandwidth 200Gbps \
  --ap_bandwidth 400Gbps \
  --nvlink_bw 2880Gbps

if [ -f "AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100" ]; then
  cp "AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100" "$OUTPUT_DIR/T2_AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100"
  rm "AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100"
  echo "✓ Saved to $OUTPUT_DIR/T2_AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100"
else
  echo "✗ Failed to generate AlibabaHPN 64 GPU"
  exit 1
fi
echo ""

# =============================================================================
# Topology 3: Spectrum-X 64 GPU (8 rails, 100Gbps) - BASELINE
# Role: Main topology for showing rail-aware co-optimization
# Key: 8 rails per GPU → high path diversity, routing matters most
# =============================================================================
echo "[3/5] Generating T3: Spectrum-X 64 GPU (100Gbps, 8 rails)..."
python gen_Topo_Template.py \
  --topology Spectrum-X \
  --ro \
  --gpu 64 \
  --gpu_per_server 8 \
  --gpu_type A100 \
  --bandwidth 100Gbps \
  --ap_bandwidth 400Gbps \
  --nvlink_bw 2880Gbps

if [ -f "Spectrum-X_64g_8gps_100Gbps_A100" ]; then
  cp "Spectrum-X_64g_8gps_100Gbps_A100" "$OUTPUT_DIR/T3_Spectrum-X_64g_8gps_100Gbps_A100"
  rm "Spectrum-X_64g_8gps_100Gbps_A100"
  echo "✓ Saved to $OUTPUT_DIR/T3_Spectrum-X_64g_8gps_100Gbps_A100"
else
  echo "✗ Failed to generate Spectrum-X 64 GPU (100Gbps)"
  exit 1
fi
echo ""

# =============================================================================
# Topology 4: Spectrum-X 64 GPU (8 rails, 200Gbps) - HIGH BANDWIDTH VARIANT
# Role: Test impact of higher bandwidth on co-optimization
# Key: Same 64 GPUs, but 2× NIC bandwidth → routing decisions change
# =============================================================================
echo "[4/5] Generating T4: Spectrum-X 64 GPU (200Gbps, 8 rails)..."
python gen_Topo_Template.py \
  --topology Spectrum-X \
  --ro \
  --gpu 64 \
  --gpu_per_server 8 \
  --gpu_type A100 \
  --bandwidth 200Gbps \
  --ap_bandwidth 800Gbps \
  --nvlink_bw 2880Gbps

if [ -f "Spectrum-X_64g_8gps_200Gbps_A100" ]; then
  cp "Spectrum-X_64g_8gps_200Gbps_A100" "$OUTPUT_DIR/T4_Spectrum-X_64g_8gps_200Gbps_A100"
  rm "Spectrum-X_64g_8gps_200Gbps_A100"
  echo "✓ Saved to $OUTPUT_DIR/T4_Spectrum-X_64g_8gps_200Gbps_A100"
else
  echo "✗ Failed to generate Spectrum-X 64 GPU (200Gbps)"
  exit 1
fi
echo ""

# =============================================================================
# Topology 5: Spectrum-X 64 GPU (8 rails, 50Gbps) - LOW BANDWIDTH VARIANT
# Role: Test co-optimization under network congestion
# Key: Same 64 GPUs, but 1/2 NIC bandwidth → congestion matters MORE
# =============================================================================
echo "[5/5] Generating T5: Spectrum-X 64 GPU (50Gbps, 8 rails)..."
python gen_Topo_Template.py \
  --topology Spectrum-X \
  --ro \
  --gpu 64 \
  --gpu_per_server 8 \
  --gpu_type A100 \
  --bandwidth 50Gbps \
  --ap_bandwidth 200Gbps \
  --nvlink_bw 2880Gbps

if [ -f "Spectrum-X_64g_8gps_50Gbps_A100" ]; then
  cp "Spectrum-X_64g_8gps_50Gbps_A100" "$OUTPUT_DIR/T5_Spectrum-X_64g_8gps_50Gbps_A100"
  rm "Spectrum-X_64g_8gps_50Gbps_A100"
  echo "✓ Saved to $OUTPUT_DIR/T5_Spectrum-X_64g_8gps_50Gbps_A100"
else
  echo "✗ Failed to generate Spectrum-X 64 GPU (50Gbps)"
  exit 1
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================="
echo "✅ All topologies generated successfully!"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Generated topologies:"
ls -lh "$OUTPUT_DIR"/T*_* 2>/dev/null || echo "(listing topology files)"
echo ""
echo "Topology Summary (all 64 GPUs):"
echo "  T1: DCN+ 64G (100Gbps, Single-ToR)      - Architecture contrast, bottleneck"
echo "  T2: AlibabaHPN 64G (200Gbps, Dual-ToR)  - Production cloud"
echo "  T3: Spectrum-X 64G (100Gbps, 8 rails)   - Baseline evaluation"
echo "  T4: Spectrum-X 64G (200Gbps, 8 rails)   - High bandwidth variant"
echo "  T5: Spectrum-X 64G (50Gbps, 8 rails)    - Low bandwidth, congested"
echo ""
echo "Expected results across topologies:"
echo "  • T1 (DCN+ bottleneck): 48% avg (placement/scheduling matter more)"
echo "  • T2 (AlibabaHPN): 50% avg (asymmetry makes coordination critical)"
echo "  • T3 (100Gbps baseline): 50% avg improvement"
echo "  • T4 (200Gbps high BW): 47% avg (bandwidth reduces routing criticality)"
echo "  • T5 (50Gbps congested): 53% avg (congestion amplifies co-opt benefits) ⭐"
echo ""
echo "Next step - Generate workloads:"
echo "  cd ../../ && ./generate_coopt_workloads.sh"
echo ""

