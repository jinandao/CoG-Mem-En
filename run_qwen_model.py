import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
import argparse

def load_models(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True, 
    )
    return base_model, tokenizer

def read_file_content(path: str) -> str:
    """读取文本文件内容"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

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

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.5
TOP_P = 0.9
DO_SAMPLE = True

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
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码生成的 tokens，仅保留新生成部分
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Qwen Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--base_instruction_dir', type=str, required=True)
    parser.add_argument('--zero_knowledge_dir', type=str, required=True)
    parser.add_argument('--composite_dir', type=str, required=True)
    args = parser.parse_args()
    model_path = args.model_path
    
    model, tokenizer = load_models(model_path)

    PROMPT_FILE = args.prompt_file

    # 2. 读取系统提示前缀
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"提示词文件不存在: {PROMPT_FILE}")
    system_prefix = read_file_content(PROMPT_FILE)

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