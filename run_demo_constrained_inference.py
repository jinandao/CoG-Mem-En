import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import json
from typing import List, Dict, Any, Optional
import re
from peft import PeftModel
import argparse

def load_models(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model = PeftModel.from_pretrained(base_model, lora_path, adapter_name="conversation")
    return model, tokenizer

def load_conversations(demo_dir: str) -> Optional[List[Dict[str, str]]]:
    """加载指定 Demo 文件夹中的 conversations.json"""
    json_path = os.path.join(demo_dir, "conversations.json")
    if not os.path.exists(json_path):
        print(f"警告: 文件不存在 {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("conversations", [])
    except Exception as e:
        print(f"读取 JSON 失败 {json_path}: {e}")
        return None

def format_conversation(convs: List[Dict[str, str]], num_turns: int = 5) -> str:
    """将前 num_turns 条对话格式化为文本"""
    text = ""
    for turn in convs[:num_turns]:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        text += "<|im_start|>"+role+"\n"+content+"<|im_end|>\n"
    text += "<|im_start|>assistant"
    return text

MAX_NEW_TOKENS = 512

# ==================== 新增：本地模型生成函数 ====================
def generate_response_with_local_model(
    model,
    tokenizer,
    text,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.1,
    top_p: float = 3,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码生成的 tokens，仅保留新生成部分
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Qwen Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--base_instruction_dir', type=str, required=True)
    parser.add_argument('--zero_knowledge_dir', type=str, required=True)
    parser.add_argument('--composite_dir', type=str, required=True)
    parser.add_argument('--partial_fallback_dir', type=str, required=True)
    args = parser.parse_args()
    model_path = args.model_path
    lora_path = args.lora_path

    model, tokenizer = load_models(model_path, lora_path)
    model.set_adapter("conversation")

    system_prefix = "<|im_start|>system\nYou are an AI assistant. When chatting with the user, if the user mentions something from the past, you need to recall by calling the function memory_query_call and passing in the retrieval query. The content of the query must include a part that describes the time and a part that contains key semantic information. Then, based on the memory returned by the memory_query role, think about the memory fragments within the `<think></think>` block, and generate the correct response based on the sorted information. If the user does not mention anything from the past, generate the correct response according to the current context.<|im_end|>\n"

    # 遍历 base_dir 下的所有子目录
    for i in range(1, 61):
        demo_name = f"Demo{i}"
        # DEMO_BASE_DIR = "./Configs/case_basic_instruction"
        DEMO_BASE_DIR = args.base_instruction_dir
        demo_path = os.path.join(DEMO_BASE_DIR, demo_name)
        print(f"\n{'=' * 40}\n处理 {demo_name}\n{'=' * 40}")

        # 加载对话数据
        convs = load_conversations(demo_path)
        text = format_conversation(convs)
        text = system_prefix + text
        response = generate_response_with_local_model(model, tokenizer, text)
        print(response)
        print("--------------------")

    for i in range(1, 31):
        demo_name = f"Demo{i}"
        # DEMO_BASE_DIR = "./Configs/case_zero_knowledge"
        DEMO_BASE_DIR = args.zero_knowledge_dir
        demo_path = os.path.join(DEMO_BASE_DIR, demo_name)
        print(f"\n{'=' * 40}\n处理 {demo_name}\n{'=' * 40}")

        # 加载对话数据
        convs = load_conversations(demo_path)
        text = format_conversation(convs)
        text = system_prefix + text
        response = generate_response_with_local_model(model, tokenizer, text)
        print(response)
        print("--------------------")

    for i in range(1, 11):
        demo_name = f"Demo{i}"
        # DEMO_BASE_DIR = "./Configs/case_zero_knowledge"
        DEMO_BASE_DIR = args.partial_fallback_dir
        demo_path = os.path.join(DEMO_BASE_DIR, demo_name)
        print(f"\n{'=' * 40}\n处理 {demo_name}\n{'=' * 40}")

        # 加载对话数据
        convs = load_conversations(demo_path)
        text = format_conversation(convs)
        text = system_prefix + text
        response = generate_response_with_local_model(model, tokenizer, text)
        print(response)
        print("--------------------")

    for i in range(1, 25):
        demo_name = f"Demo{i}"
        # DEMO_BASE_DIR = "./Configs/case_composite"
        DEMO_BASE_DIR = args.composite_dir
        demo_path = os.path.join(DEMO_BASE_DIR, demo_name)
        print(f"\n{'=' * 40}\n处理 {demo_name}\n{'=' * 40}")

        # 加载对话数据
        convs = load_conversations(demo_path)
        text = format_conversation(convs)
        text = system_prefix + text
        response = generate_response_with_local_model(model, tokenizer, text)
        print(response)
        print("--------------------")