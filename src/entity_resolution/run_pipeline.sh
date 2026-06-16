#!/bin/bash
# Entity Resolution Pipeline
# Usage: bash run_pipeline.sh <prefix> <threshold> [gpu]
# Example: bash run_pipeline.sh openai 0.6547        # GPU 0 (default)
#          bash run_pipeline.sh qwen32b 0.6627 1     # pin to physical GPU 1

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash run_pipeline.sh <prefix> <threshold> [gpu]"
    echo "  prefix    : dataset prefix, e.g. openai or qwen32b"
    echo "  threshold : calibrated FAISS floor, e.g. 0.6547"
    echo "  gpu       : physical GPU id to pin (PCI-bus order, matches nvidia-smi); default 0"
    exit 1
fi

PREFIX=$1
THRESHOLD=$2
GPU=${3:-0}

echo "====================================================="
echo "ENTITY RESOLUTION PIPELINE: $PREFIX"
echo "Threshold: $THRESHOLD | GPU: $GPU"
echo "====================================================="

# Step 0: Normalize dates (run once after extraction, before ER)
echo -e "\n[Step 0] Normalizing date fields..."
python normalize_dates.py \
    --entities  ${PREFIX}_entities_final.csv \
    --relations ${PREFIX}_relations_final.csv \
    --output    ${PREFIX}_entities_normdates.csv

# Step 1: FAISS Blocking
echo -e "\n[Step 1/4] FAISS Blocking..."
python faiss_blocking.py \
    --entities  ${PREFIX}_entities_normdates.csv \
    --relations ${PREFIX}_relations_final.csv \
    --output    ${PREFIX}_grey_zone.csv \
    --threshold $THRESHOLD \
    --top_k 10 \
    --gpu $GPU \
    --log ${PREFIX}_step1_faiss.log

# Step 2: LLM Resolution (Qwen2.5-32B via vLLM)
echo -e "\n[Step 2/4] LLM Resolution..."
python llm_judgement.py \
    --grey_zone  ${PREFIX}_grey_zone.csv \
    --golden_set ${PREFIX}_golden_entity_pairs.csv \
    --output     ${PREFIX}_llm_resolved.csv \
    --model      Qwen/Qwen2.5-32B-Instruct \
    --chunk_size 48 \
    --gpu $GPU \
    --log ${PREFIX}_step2_vllm.log

# Step 3: Graph Node Fusion (includes person property patching)
echo -e "\n[Step 3/4] Graph Node Fusion..."
python node_fusion.py \
    --llm_resolved    ${PREFIX}_llm_resolved.csv \
    --raw_entities    ${PREFIX}_entities_normdates.csv \
    --raw_relations   ${PREFIX}_relations_final.csv \
    --fused_entities  ${PREFIX}_entities_fused.csv \
    --fused_relations ${PREFIX}_relations_fused.csv \
    --log ${PREFIX}_step3_fusion.log

# Step 4: Evaluation
echo -e "\n[Step 4/4] Pipeline Evaluation..."
python evaluate_pipeline.py \
    --golden_set    ${PREFIX}_golden_entity_pairs.csv \
    --llm_resolved  ${PREFIX}_llm_resolved.csv \
    --grey_zone     ${PREFIX}_grey_zone.csv \
    --errors_output ${PREFIX}_pipeline_errors.csv \
    --log ${PREFIX}_step4_eval.log

echo -e "\nPIPELINE COMPLETE FOR $PREFIX"
echo "Check ${PREFIX}_step4_eval.log for final metrics."
echo "====================================================="
