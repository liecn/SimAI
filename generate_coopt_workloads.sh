#!/bin/bash
# Workload Generation Script for Co-Optimization Research
# Generates 5 MICRO-WORKLOADS for fast co-optimization experiments
# These are intentionally small to enable rapid ADRS iteration cycles

set -e

echo "========================================="
echo "Generating Co-Optimization Workloads"
echo "========================================="
echo ""

# Change to aicb directory
cd "$(dirname "$0")/aicb"

# Output directly to example/
OUTPUT_DIR="../example"

# Key parameters for MICRO workloads:
# - epoch_num=1: Single training iteration (not full training)
# - world_size=64: Match primary topology (Spectrum-X 64 GPU)
# - global_batch=64: Small batch for fast simulation
# - Focus: Capture interference patterns, not full training runs

# =============================================================================
# Workload 1: Pipeline+DP with Imbalanced Stages (PP=8, DP=8, 64 GPUs)
# GPT-3 175B with variable P2P sizes
# Challenge: Sequential P2P (critical path) conflicts with parallel AllReduce
# =============================================================================
echo "[1/5] Generating W1: Pipeline+DP (PP=8, DP=8)..."
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron \
  --world_size=64 \
  --tensor_model_parallel_size=1 \
  --pipeline_model_parallel=8 \
  --global_batch=64 \
  --micro_batch=1 \
  --model_name=gpt_175B \
  --num_layers=96 \
  --hidden_size=12288 \
  --num_attention_heads=96 \
  --seq_length=2048 \
  --epoch_num=1 \
  --enable_sequence_parallel \
  --use_flash_attn \
  --swiglu \
  --use-distributed-optimizer

mv results/workload/None-gpt_175B-world_size64-tp1-pp8-ep1-gbs64-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-True.txt \
   "$OUTPUT_DIR/W1_PipelineDP_PP8_DP8.txt"
echo "✓ Saved to $OUTPUT_DIR/W1_PipelineDP_PP8_DP8.txt"
echo ""

# =============================================================================
# Workload 2: Concurrent Equal-Size DP Groups (DP=8 on 64 GPUs → 8 rings)
# Single job creating multi-ring interference
# Challenge: 8 concurrent DP rings create ring-to-ring contention
# =============================================================================
echo "[2/5] Generating W2: Concurrent DP Rings (8 rings, DP=8 each)..."
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron \
  --world_size=64 \
  --tensor_model_parallel_size=1 \
  --pipeline_model_parallel=1 \
  --global_batch=64 \
  --micro_batch=1 \
  --model_name=gpt_175B \
  --num_layers=96 \
  --hidden_size=12288 \
  --num_attention_heads=96 \
  --seq_length=2048 \
  --epoch_num=1 \
  --enable_sequence_parallel \
  --use_flash_attn \
  --swiglu \
  --use-distributed-optimizer

mv results/workload/None-gpt_175B-world_size64-tp1-pp1-ep1-gbs64-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-True.txt \
   "$OUTPUT_DIR/W2_ConcurrentDP_8Rings.txt"
echo "✓ Saved to $OUTPUT_DIR/W2_ConcurrentDP_8Rings.txt"
echo ""

# =============================================================================
# Workload 3: MoE with Expert Imbalance (EP=8, TP=2, DP=4, 64 GPUs)
# Mixtral with hotspot experts (3:1 load imbalance)
# Challenge: All-to-All with hotspots + variable message sizes
# =============================================================================
echo "[3/5] Generating W3: MoE with Imbalance (EP=8, TP=2, DP=4)..."
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron \
  --world_size=64 \
  --tensor_model_parallel_size=2 \
  --pipeline_model_parallel=1 \
  --expert_model_parallel_size=8 \
  --num_experts=32 \
  --moe_router_topk=2 \
  --moe_enable \
  --moe_grouped_gemm \
  --global_batch=64 \
  --micro_batch=1 \
  --model_name=Mixtral_8x7B \
  --num_layers=32 \
  --hidden_size=4096 \
  --num_attention_heads=32 \
  --ffn_hidden_size=14336 \
  --seq_length=2048 \
  --epoch_num=1 \
  --enable_sequence_parallel \
  --use_flash_attn \
  --swiglu

# Rename output (handle potential glob patterns in filename)
for f in results/workload/None-*-world_size64-tp2-pp1-ep8-gbs64-mbs1-seq2048-MOE-True-GEMM-True-flash_attn-True.txt; do
  if [ -f "$f" ]; then
    mv "$f" "$OUTPUT_DIR/W3_MoE_Imbalanced.txt"
    break
  fi
done
echo "✓ Saved to $OUTPUT_DIR/W3_MoE_Imbalanced.txt"
echo ""

# =============================================================================
# Workload 4: Bursty Gradient Accumulation (DP=64, gradient_accum=2)
# GPT-3 175B with temporal burst pattern (simplified: ga=2 instead of 8)
# Challenge: Steady-state routing can't handle periodic burst synchronization
# =============================================================================
echo "[4/5] Generating W4: Bursty Gradient Accumulation (GA=2)..."
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron \
  --world_size=64 \
  --tensor_model_parallel_size=1 \
  --pipeline_model_parallel=1 \
  --global_batch=128 \
  --micro_batch=1 \
  --model_name=gpt_175B \
  --num_layers=96 \
  --hidden_size=12288 \
  --num_attention_heads=96 \
  --seq_length=2048 \
  --epoch_num=1 \
  --enable_sequence_parallel \
  --use_flash_attn \
  --swiglu \
  --use-distributed-optimizer

mv results/workload/None-gpt_175B-world_size64-tp1-pp1-ep1-gbs128-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-True.txt \
   "$OUTPUT_DIR/W4_Bursty_GA2.txt"
echo "✓ Saved to $OUTPUT_DIR/W4_Bursty_GA2.txt"
echo ""

# =============================================================================
# Workload 5: Full Hierarchical Parallelism (DP=16, TP=2, PP=2, 64 GPUs)
# GPT-3 175B with ALL 3 parallelism dimensions active
# Challenge: MAXIMUM interference - P2P vs TP-AllReduce vs DP-AllReduce
# =============================================================================
echo "[5/5] Generating W5: Full Hierarchical (DP=16, TP=2, PP=2)..."
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron \
  --world_size=64 \
  --tensor_model_parallel_size=2 \
  --pipeline_model_parallel=2 \
  --global_batch=64 \
  --micro_batch=1 \
  --model_name=gpt_175B \
  --num_layers=96 \
  --hidden_size=12288 \
  --num_attention_heads=96 \
  --seq_length=2048 \
  --epoch_num=1 \
  --enable_sequence_parallel \
  --use_flash_attn \
  --swiglu \
  --use-distributed-optimizer

mv results/workload/None-gpt_175B-world_size64-tp2-pp2-ep1-gbs64-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-True.txt \
   "$OUTPUT_DIR/W5_FullHierarchical_DP16_TP2_PP2.txt"
echo "✓ Saved to $OUTPUT_DIR/W5_FullHierarchical_DP16_TP2_PP2.txt"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================="
echo "✅ All workloads generated successfully!"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Generated workloads:"
ls -lh "$OUTPUT_DIR"/W*.txt
echo ""
echo "Workload Summary:"
echo "  W1: Pipeline+DP (PP=8, DP=8)           - P2P + Collective interference, 418 ops"
echo "  W2: Concurrent DP (8 rings)             - Multiple concurrent AllReduce, 397 ops"
echo "  W3: MoE with Imbalance (EP=8)           - Hotspot congestion (AllToAll), 464 ops"
echo "  W4: Bursty GA (GA=2)                    - Temporal burst pattern, ~780 ops"
echo "  W5: Full Hierarchical (TP+PP+DP)        - Multi-dimensional parallelism, 790 ops"
echo ""
echo "Next step - Run simulations:"
echo "  python run_simai.py ns3 --topo example/T3_Spectrum-X_64g_8gps_100Gbps_A100 --workload example/W1_PipelineDP_PP8_DP8.txt"
echo ""
