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


def load_models(model_path, compress_model_path, query_model_path, conversation_model_path):
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
    model = PeftModel.from_pretrained(base_model, compress_model_path, adapter_name="compress")
    model.load_adapter(query_model_path, adapter_name="query")
    model.load_adapter(conversation_model_path, adapter_name="conversation")
    return model, tokenizer

def load_data(conversation_path, conversation_teach_path, memories_path):
    # load conversation.json
    with open(conversation_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)
    # load conversation_teach.json
    with open(conversation_teach_path, 'r', encoding='utf-8') as f:
        conversation_teach_data = json.load(f)
    # load memories.json
    with open(memories_path, 'r', encoding='utf-8') as f:
        memories_data = json.load(f)
    return conversation_data, conversation_teach_data, memories_data


def compress_data(compress_model, conversation):
    model.set_adapter("compress")
    conversations = conversation['conversation']
    whole_str = f"<|im_start|>system\nYou are an AI assistant skilled at summarizing and condensing conversations. You need to extract and sort out the key points of the conversation in the <think></think> blocks, and then write a final summary in the <memory></memory> section."
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n"+ conversations[i]['content'] + "<|im_end|>\n"
        whole_str += cur_input_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(compress_model.device)
    outputs = compress_model.generate(**inputs, max_new_tokens=1024,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    whole_str += response
    return response

def get_azy_timestamp():
    """生成符合 Azy-Memory 系统标准的格式化时间戳"""
    return datetime.now().strftime("%Y-%m-%d-%H:%M")

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

def query_data(model, memories, query):
    model.set_adapter("query")
    whole_str =  f"<|im_start|>system\nYou are an AI assistant skilled in retrieving relevant memories from the memory bank based on the query. You need to indicate highly relevant memories in the <related_memories></related_memories> section and lowly relevant memories in the <low_related_memories></low_related_memories> section."
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(
            memory_content) + "<|im_end|>"
        whole_str += cur_input_str

    query_str = "\n<|im_start|>query\n" + query + "<|im_end|>"
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
            mem_item_str = "[mem - id: " + str(id) + "] Time:" + memory["time"] + ", Content: " + memory["memory"]
            memory_str += mem_item_str
    if len(results['low_related_memories']) > 0:
        memory_str = "Low-relevance memory snippets: "
        low_related_memories_ids = results['low_related_memories']
        for id in low_related_memories_ids:
            memory = memories[id - 1]
            low_mem_item_str = "[mem - id: " + str(id) + "] Time:" + memory["time"] + ", Content:" + memory["memory"]
            memory_str += low_mem_item_str
    return memory_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--compress_model_path', type=str, required=True, help='Path to the conversation compression model')
    parser.add_argument('--retrieval_model_path', type=str, required=True, help='Path to the memory query model')
    parser.add_argument('--synthesis_model_path', type=str, required=True, help='Path to the conversation generation model')
    parser.add_argument('--conversation_path', type=str, required=True, help='Path to the conversation file')
    parser.add_argument('--conversation_teach_path', type=str, required=True, help='Path to the teaching conversation file')
    parser.add_argument('--memories_path', type=str, required=True, help='Path to the memories file')

    args = parser.parse_args()

    model, tokenizer = load_models(args.model_path, args.compress_model_path, args.retrieval_model_path, args.synthesis_model_path)
    conversation_data, conversation_teach_data, memories_data = load_data(args.conversation_path, args.conversation_teach_path, args.memories_path)
    memories = memories_data["memories"]

    # compress conversation to memory
    add_memory = compress_data(model, conversation_teach_data)
    print("compress memory:", add_memory)
    print("------------------------------")
    match = re.search(r'<memory>(.*?)</memory>', add_memory, re.S)
    memory_content = ""
    if match:
        memory_content = match.group(1).strip()

    # add memory to memories
    time_str = get_azy_timestamp()
    new_memory = {"mem_id": len(memories) + 1, "time": time_str, "memory": memory_content}
    memories.append(new_memory)
    print("all memory items:")
    for memory in memories:
        print(memory)
    print("------------------------------")

    messages = conversation_data['conversations']
    whole_str = f"<|im_start|>system\nYou are an AI assistant having a conversation with a user. When the user mentions things from the past, you need to call the memory_query_call function and pass the query as the parameter to retrieve memories. Then, based on the memories returned by the memory_query role, think through the information of the memory fragments within the blocks, and generate a proper response according to the organized information. If the user does not mention anything from the past, generate a proper response based on the current context."
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
    print("original conversation:", whole_str)
    print("------------------------------")
    print("start generation")

    messages = conversation_data['conversations']
    whole_str = f"<|im_start|>system\nYou are an AI assistant having a conversation with a user. When the user mentions things from the past, you need to call the memory_query_call function and pass the query as the parameter to retrieve memories. Then, based on the memories returned by the memory_query role, think through the information of the memory fragments within the blocks, and generate a proper response according to the organized information. If the user does not mention anything from the past, generate a proper response based on the current context."
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
                    related_memories = query_data(model, memories, query_content)
                    print("Retrieved memories:", related_memories, " Relevant query:", query_content)
                    print("------------------------------")
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

