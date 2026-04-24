MODEL_PATH="Qwen/Qwen3-8B"

COMPRESS_PATH="jinandao/memory_compress_lora"
QUERY_PATH="jinandao/memory_query_lora"
CONVERSATIONS_PATH="jinandao/memory_conversation_lora"
BASE_DIR="./Configs/case_basic_instruction"
MEMORIES_FILE="./Configs/query_normal_en.json"
# ===============================

echo "Start Performing..."
python run_demo_en_basic_instruction.py \
    --model_path "$MODEL_PATH" \
    --compress_model_path "$COMPRESS_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_DIR" \
    --memories_path "$MEMORIES_FILE"

BASE_ZERO_KNOWLEDGE_DIR="./Configs/case_zero_knowledge"
python run_demo_en_zero_knowledge.py \
    --model_path "$MODEL_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_ZERO_KNOWLEDGE_DIR" \
    --memories_path "$MEMORIES_FILE"

BASE_PARTIAL_FALLBACK_DIR="./Configs/case_partial_fallback"
python run_demo_en_partial_fallback.py \
    --model_path "$MODEL_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_PARTIAL_FALLBACK_DIR" 

BASE_COMPOSITE_DIR="./Configs/case_composite"
python run_demo_en_composite.py \
    --model_path "$MODEL_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_COMPOSITE_DIR" 