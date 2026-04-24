MODEL_PATH="Qwen/Qwen3-8B"
PROMPT_FILE="./Configs/qwen_prompts/complex_prompts.txt"
BASE_INSTRUCTION_DIR="./Configs/case_basic_instruction"
ZERO_KNOWLEDGE_DIR="./Configs/case_zero_knowledge"
PARTIAL_FALLBACK_DOR="./Configs/case_partial_fallback"
COMPOSITE_DIR="./Configs/case_composite"

python run_qwen_model.py \
    --model_path "$MODEL_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --base_instruction_dir "$BASE_INSTRUCTION_DIR" \
    --zero_knowledge_dir "$ZERO_KNOWLEDGE_DIR" \
    --composite_dir "$COMPOSITE_DIR" \
    --partial_fallback_dir "$PARTIAL_FALLBACK_DOR" 