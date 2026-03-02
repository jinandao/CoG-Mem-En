#!/bin/bash

# Set params
TRAIN_JSON_PATH="./Datasets/memory_synthesis/conversation_use_memory_en.json"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SEED=12345
OUTPUT_DIR="./Output/Memory_Synthesis/Conversation_Use_Memory_En"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STEPS=10
NUM_TRAIN_EPOCHS=16
LEARNING_RATE=2e-5

# Start training
python memory_synthesis_sft.py \
    --train_json_path "$TRAIN_JSON_PATH" \
    --model_dir "$MODEL_DIR" \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps $LOGGING_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_checkpointing