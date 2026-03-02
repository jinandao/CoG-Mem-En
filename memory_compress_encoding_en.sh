#!/bin/bash

# ==========================================
# Course Learning: Compression Conversation Training Script
# 1. First run SFT training
# 2. Then run DPO training (using the SFT-trained model)
# ==========================================

# Set common parameters
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"

# SFT training parameters
SFT_TRAIN_JSON_PATH="./Datasets/compress_conversation/compress_conversation_sft_en.json"
SFT_OUTPUT_DIR="./Output/Compress_Conversation/Compress_Conversation_SFT_en"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE=2
SFT_GRADIENT_ACCUMULATION_STEPS=1
SFT_NUM_TRAIN_EPOCHS=4
SFT_LEARNING_RATE=2e-5

# DPO training parameters
DPO_TRAIN_JSON_PATH="./Datasets/compress_conversation/compress_conversation_dpo_en.json"
DPO_OUTPUT_DIR="./Output/Compress_Conversation/Compress_Conversation_DPO_en"
DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
DPO_GRADIENT_ACCUMULATION_STEPS=2
DPO_LEARNING_RATE=2e-6
DPO_BETA=0.5
DPO_MAX_LENGTH=4096
DPO_MAX_PROMPT_LENGTH=2560
DPO_MAX_GRAD_NORM=0.1
DPO_NUM_TRAIN_EPOCHS=1
DPO_SEED=150

# ==========================================
# Step 1: Run SFT Training
# ==========================================
echo "start SFT training..."
echo "SFT path: $SFT_OUTPUT_DIR"

python memory_compress_encoding_sft.py \
    --train_json_path "$SFT_TRAIN_JSON_PATH" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --per_device_train_batch_size $SFT_PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $SFT_GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $SFT_NUM_TRAIN_EPOCHS \
    --learning_rate $SFT_LEARNING_RATE \
    --gradient_checkpointing

SFT_EXIT_CODE=$?
if [ $SFT_EXIT_CODE -ne 0 ]; then
    echo "SFT train failed: $SFT_EXIT_CODE"
    exit 1
fi

echo "SFT train success！"
# Option A: If code was modified according to Option 1, the model will be saved in the final_model subdirectory
SFT_MODEL_PATH=""
FINAL_MODEL_DIR="$SFT_OUTPUT_DIR/final_model"
if [ -d "$FINAL_MODEL_DIR" ]; then
    SFT_MODEL_PATH="$FINAL_MODEL_DIR"
    echo "find final_model path: $SFT_MODEL_PATH"
else
    SFT_MODEL_PATH="$SFT_OUTPUT_DIR"
    echo "not find final_model path, SFT Path: $SFT_MODEL_PATH"
fi
# ==========================================
# Step 2: Run DPO Training
# Using the SFT-trained model as the starting point
# ==========================================
echo ""
echo "start DPO training..."
echo "SFT model dir: $SFT_OUTPUT_DIR"
echo "DPO output dir: $DPO_OUTPUT_DIR"

python memory_compress_encoding_dpo.py \
    --model_dir "$MODEL_DIR" \
    --train_json_path "$DPO_TRAIN_JSON_PATH" \
    --sft_lora_path "$SFT_MODEL_PATH" \
    --per_device_train_batch_size $DPO_PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $DPO_GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $DPO_LEARNING_RATE \
    --output_dir "$DPO_OUTPUT_DIR" \
    --beta $DPO_BETA \
    --max_length $DPO_MAX_LENGTH \
    --max_prompt_length $DPO_MAX_PROMPT_LENGTH \
    --max_grad_norm $DPO_MAX_GRAD_NORM \
    --num_train_epochs $DPO_NUM_TRAIN_EPOCHS \
    --seed $DPO_SEED \
    --gradient_checkpointing \
    --fp16 \
    --continue_train

DPO_EXIT_CODE=$?
if [ $DPO_EXIT_CODE -ne 0 ]; then
    echo "DPO train failed: $DPO_EXIT_CODE"
    exit 1
fi