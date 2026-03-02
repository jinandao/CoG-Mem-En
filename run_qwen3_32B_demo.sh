#!/bin/bash
# Qwen3-32B-FP8 Azy-Memory Demo Script (Hardcoded Parameters Version)

# ==== Please modify your paths here ====
MODEL_PATH="Qwen/Qwen3-32B-FP8"

# ===============================

echo "Start Qwem3-32B Azy-Memory Performing..."
python run_qwen3_32B_demo.py \
    --model_path "$MODEL_PATH" \
