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
from peft import PeftModel
import json
from datetime import datetime
import re
import argparse
import random
from datetime import datetime, timedelta


def load_models(model_path, query_model_path, conversation_model_path):
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
    model = PeftModel.from_pretrained(base_model, query_model_path, adapter_name="query")
    model.load_adapter(conversation_model_path, adapter_name="conversation")
    return model, tokenizer

def load_data(conversation_path):
    # load conversation.json
    with open(conversation_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)
    return conversation_data


def generate_memories(normal_memories):
    num_to_sample = 30
    if num_to_sample > 0 and len(normal_memories) > 0:
        sampled_normal = random.sample(
            normal_memories,
            min(num_to_sample, len(normal_memories))
        )
    else:
        sampled_normal = []
    random.shuffle(sampled_normal)

    memories_with_id = []
    for idx, memory in enumerate(sampled_normal, 1):
        memories_with_id.append({
            "mem_id": idx,
            "memory": memory['memory']
        })
    time_list = []
    start_year = random.choice([2025, 2026])
    start_month = random.randint(1, 12)
    start_day = random.randint(1, 28)
    base_hour = random.randint(12, 20)
    base_minute = random.randint(0, 59)
    current_time = datetime(start_year, start_month, start_day, base_hour, base_minute)
    interval_range = (6, 9)
    for i in range(len(sampled_normal)):
        if i == 0:
            time_list.append(current_time.strftime("%Y-%m-%d-%H:%M"))
        else:
            interval_hours = random.randint(interval_range[0], interval_range[1])
            current_time = current_time + timedelta(hours=interval_hours)
            time_list.append(current_time.strftime("%Y-%m-%d-%H:%M"))
    new_memories = []
    for i, mem in enumerate(memories_with_id):
        new_mem = {
            "mem_id": mem['mem_id'],
            "memory": mem['memory'],
            "time": time_list[i]
        }
        new_memories.append(new_mem)
    return new_memories, current_time

def generate_reference_data(conversation_data):
    messages = conversation_data['conversations']
    whole_str = f"<|im_start|>system\nYou are an AI assistant. When chatting with the user, if the user mentions something from the past, you need to recall by calling the function memory_query_call and passing in the retrieval query. The content of the query must include a part that describes the time and a part that contains key semantic information. Then, based on the memory returned by the memory_query role, think about the memory fragments within the `<think></think>` block, and generate the correct response based on the sorted information. If the user does not mention anything from the past, generate the correct response according to the current context."
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i][
                'content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
        else:
            cur_input_str = ""
            if 'think' in messages[i]:
                cur_input_str += "<think>" + messages[i]['think'] + "</think>"
            cur_input_str += messages[i]['content']
            whole_str += cur_input_str
    return whole_str

def parse_memory_id(text):
    tags = ['related_memories', 'low_related_memories']
    results = {}
    for tag in tags:
        # 1. Extract the content between the tags
        match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.S)
        if match:
            content = match.group(1).strip()
            # 2. Determine if it is the string "None"
            if content.lower() == 'none' or not content:
                results[tag] = []
                continue
            # 3. Attempt to parse as an array
            try:
                # Use json.loads to convert "[5]" to [5]
                data = json.loads(content)
                if isinstance(data, list):
                    # Ensure only numbers are taken (filter non-numeric items)
                    results[tag] = [int(x) for x in data if isinstance(x, (int, float))]
                else:
                    results[tag] = []
            except (json.JSONDecodeError, ValueError):
                # If not standard JSON format, try to extract numbers with regex (error-tolerant handling)
                nums = re.findall(r'\d+', content)
                results[tag] = [int(n) for n in nums]
        else:
            results[tag] = []
    return results

def query_data(model, memories, query_time, query):
    model.set_adapter("query")
    whole_str =  f"<|im_start|>system\nYou are an AI assistant skilled at searching for relevant memories in the memories section based on a query. You need to list highly relevant memories within <related_memories></related_memories> and low-relevance memories within <low_related_memories></low_related_memories>."
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(
            memory_content) + "<|im_end|>"
        whole_str += cur_input_str
    query_str = "<|im_start|>query\n" + "time:" + str(query_time) + "\ncontent:" + query + "<|im_end|>"
    whole_str += query_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    results = parse_memory_id(response)
    memory_str = "Relevant memory snippets: "
    if len(results['related_memories']) > 0:
        related_memories_ids = results['related_memories']
        for id in related_memories_ids:
            memory = memories[id - 1]
            mem_item_str = "[mem - id: " + str(id) + "] Time:" + str(memory["time"]) + ", Content: " + memory["memory"]
            memory_str += mem_item_str
    else:
        memory_str += "None."
    memory_str += " Low relevance memory fragments: "
    if len(results['low_related_memories']) > 0:
        low_related_memories_ids = results['low_related_memories']
        for id in low_related_memories_ids:
            memory = memories[id - 1]
            low_mem_item_str = "[mem - id: " + str(id) + "] Time:" + memory["time"] + ", Content:" + memory["memory"]
            memory_str += low_mem_item_str
    else:
        memory_str += "None"
    return memory_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--query_model_path', type=str, required=True, help='Path to the memory query model')
    parser.add_argument('--conversation_model_path', type=str, required=True, help='Path to the conversation generation model')
    parser.add_argument('--memories_path', type=str, required=True, help='Path to the memories file')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the base file')
    args = parser.parse_args()

    model_path = args.model_path
    
    query_model_path = args.query_model_path
    conversation_model_path = args.conversation_model_path
    memories_path = args.memories_path
    base_dir = args.base_dir
    model, tokenizer = load_models(model_path, query_model_path, conversation_model_path)

    with open(memories_path, 'r', encoding='utf-8') as f:
        normal_memories = json.load(f)
    
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    folders_sorted = sorted(folders, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

    # 遍历 base_dir 下的所有子目录
    for i in range(0, len(folders_sorted)):
        folder_name = folders_sorted[i]
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        print("folder_name:", folder_name)
        conversation_path = os.path.join(folder_path, "conversations.json")
        conversation_data = load_data(conversation_path)

        print("---------------")

        memories, last_time = generate_memories(normal_memories)
        query_time = last_time + timedelta(hours=6)
        # for memory in memories:
        #     print("memory:", memory)
        # print("---------------")

        whole_str = generate_reference_data(conversation_data)
        print("reference conversation:", whole_str)
        print("---------------")
        print("start generation")

        messages = conversation_data['conversations']
        whole_str = f"<|im_start|>system\nYou are an AI assistant. When chatting with the user, if the user mentions something from the past, you need to recall by calling the function memory_query_call and passing in the retrieval query. The content of the query must include a part that describes the time and a part that contains key semantic information. Then, based on the memory returned by the memory_query role, think about the memory fragments within the `<think></think>` block, and generate the correct response based on the sorted information. If the user does not mention anything from the past, generate the correct response according to the current context."
        for i in range(len(messages)):
            if messages[i]['role'] == 'user':
                cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i][
                    'content'] + "<|im_end|>\n<|im_start|>assistant\n"
                whole_str += cur_input_str
                inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
                model.set_adapter("conversation")
                outputs = model.generate(**inputs,
                                         max_new_tokens=384,
                                         do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id, )
                response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                whole_str += response

                func_match = re.search(r'<function>(.*?)</function>', response)
                cont_match = re.search(r'<content>(.*?)</content>', response)
                if func_match and cont_match:
                    func_type = func_match.group(1).strip()
                    query_content = cont_match.group(1).strip()
                    print(f"Executing operation: {func_type}")
                    print(f"Search content: {query_content}")
                    if func_type == "memory_query_call":
                        related_memories = query_data(model, memories, query_time, query_content)
                        print("Retrieved memories:", related_memories)
                        print("---------------")
                        cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + related_memories + "<|im_end|>\n<|im_start|>assistant\n"
                        whole_str += cur_input_str
                        inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
                        model.set_adapter("conversation")
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=384,
                                                 do_sample=False,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 eos_token_id=tokenizer.eos_token_id, )
                        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                        whole_str += response
        print("Complete generated conversation:", whole_str)
        print("---------------------------------------")
        print("---------------------------------------")