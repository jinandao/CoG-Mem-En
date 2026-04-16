#!/bin/bash

# Set params
MODEL_DIR="/root/Models/Qwen3-8B"
TRAIN_JSON_PATH="./Datasets/query_en/train/memory_query_train.json"
TEST_JSON_PATH="./Datasets/query_en/test/memory_query_test.json"
OUTPUT_DIR="./Output/Query_Memory"
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LOGGING_STEPS=10
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=2e-5

# Start training
python memory_query_sft_en.py \
    --model_dir "$MODEL_DIR" \
    --train_json_path "$TRAIN_JSON_PATH" \
    --test_json_path "$TEST_JSON_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps $LOGGING_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_checkpointing