MODEL_PATH="Qwen/Qwen3-8B"
LORA_PATH="jinandao/memory_conversation_lora"
BASE_INSTRUCTION_DIR="./Configs/case_basic_instruction"
ZERO_KNOWLEDGE_DIR="./Configs/case_zero_knowledge"
COMPOSITE_DIR="./Configs/case_composite"

python run_demo_constrained_inference.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --base_instruction_dir "$BASE_INSTRUCTION_DIR" \
    --zero_knowledge_dir "$ZERO_KNOWLEDGE_DIR" \
    --composite_dir "$COMPOSITE_DIR" 