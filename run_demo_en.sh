#!/bin/bash
# Azy-Memory Demo Script (Hardcoded Parameters Version)
# Please modify the paths below and run directly: ./run_demo_en.sh

# ==== Please modify your paths here ====
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
COMPRESS_PATH="jinandao/memory_encoding_lora"
RETRIEVAL_PATH="jinandao/memory_retrieval_lora"
SYNTHESIS_PATH="jinandao/memory_synthesis_lora"
CONVERSATION_FILE="./Configs/demo2_En/conversation.json"
TEACH_FILE="./Configs/demo2_En/conversation_teach.json"
MEMORIES_FILE="./Configs/demo2_En/memories.json"
# ===============================

echo "Start Azy-Memory Performing..."
python run_demo.py \
    --model_path "$MODEL_PATH" \
    --compress_model_path "$COMPRESS_PATH" \
    --retrieval_model_path "$RETRIEVAL_PATH" \
    --synthesis_model_path "$SYNTHESIS_PATH" \
    --conversation_path "$CONVERSATION_FILE" \
    --conversation_teach_path "$TEACH_FILE" \
    --memories_path "$MEMORIES_FILE"