MODEL_PATH="Qwen/Qwen3-32B-FP8"
PROMPT_FILE="./Configs/qwen_prompts/complext_prompts.txt"
BASE_INSTRUCTION_DIR="./Configs/case_basic_instruction"
ZERO_KNOWLEDGE_DIR="./Configs/case_zero_knowledge"
COMPOSITE_DIR="./Configs/case_composite"

python run_qwen_model.py \
    --model_path "$MODEL_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --base_instruction_dir "$BASE_INSTRUCTION_DIR" \
    --zero_knowledge_dir "$ZERO_KNOWLEDGE_DIR" \
    --composite_dir "$COMPOSITE_DIR" 