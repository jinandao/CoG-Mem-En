import json
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import os
os.environ["WANDB_DISABLED"] = "true"
import re
import argparse
from functools import partial

def process_func(example, tokenizer):
    """
    将数据集进行预处理
    """
    # print(example)
    memories = example['memories']
    input_str = f"<|im_start|>system\nYou are an AI assistant skilled at searching for relevant memories in the memories section based on a query. You need to list highly relevant memories within <related_memories></related_memories> and low-relevance memories within <low_related_memories></low_related_memories>.\n"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(memory_content) + "<|im_end|>"
        cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        labels.extend([-100] * len(cur_input_ids['input_ids']))
        input_str += cur_input_str

    query = example['query']
    query_time = example['query_time']
    query_str = "<|im_start|>query\n" + "time:" + str(query_time) + "\ncontent:"  + query + "<|im_end|>"
    query_ids = tokenizer(query_str, add_special_tokens=False)
    input_ids.extend(query_ids['input_ids'])
    attention_mask.extend(query_ids['attention_mask'])
    labels.extend([-100] * len(query_ids['input_ids']))
    input_str += query_str

    related_memories = example['related_memories']
    low_related_memories = example['low_related_memories']
    if low_related_memories is None:
        low_related_memories = []
    final_memories_str = "<related_memories>" + str(related_memories) + "</related_memories><low_related_memories>" + str(low_related_memories) + "</low_related_memories><|im_end|>"
    final_memories_ids = tokenizer(final_memories_str, add_special_tokens=False)
    input_ids.extend(final_memories_ids['input_ids'])
    attention_mask.extend(final_memories_ids['attention_mask'])
    labels.extend(final_memories_ids['input_ids'])
    input_str += final_memories_str

    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于4096的样本"""
    return example["length"] <= 4096

def check_right(related_memories, low_related_memories, response):
    try:
        related_match = re.search(r'<related_memories>\[(.*?)\]</related_memories>', response)
        low_related_match = re.search(r'<low_related_memories>\[(.*?)\]</low_related_memories>', response)
        is_related_match = False
        is_low_related_match = False
        if related_match is not None:
            related_match_split = related_match.group(1).split(',')
            if related_match_split[0] != '':
                related_numbers = [int(num.strip()) for num in related_match_split]
                if len(related_numbers) == len(related_memories):
                    is_all_equal = True
                    for i in range(len(related_numbers)):
                        if related_numbers[i] != related_memories[i]:
                            is_all_equal = False
                            break
                    if is_all_equal:
                        is_related_match = True
            else:
                if related_memories is None or len(related_memories) == 0:
                    is_related_match = True
        if low_related_match is not None:
            low_related_match_split = low_related_match.group(1).split(',')
            if low_related_match_split[0] != '':
                low_related_numbers = [int(num.strip()) for num in low_related_match_split]
                if len(low_related_numbers) == len(low_related_memories):
                    is_all_equal = True
                    for i in range(len(low_related_numbers)):
                        if low_related_numbers[i] != low_related_memories[i]:
                            is_all_equal = False
                            break
                    if is_all_equal:
                        is_low_related_match = True
            else:
                if low_related_memories is None or len(low_related_memories) == 0:
                    is_low_related_match = True
        return is_related_match and is_low_related_match
    except:
        return False


def predict(example, model, tokenizer, total_nums, right_nums):
    memories = example['memories']
    whole_str = f"<|im_start|>system\nYou are an AI assistant skilled at searching for relevant memories in the memories section based on a query. You need to list highly relevant memories within <related_memories></related_memories> and low-relevance memories within <low_related_memories></low_related_memories>.\n"
    for i in range(len(memories)):
        memory_item = memories[i]
        memory_time = memory_item["time"]
        memory_id = memory_item["mem_id"]
        memory_content = memory_item["memory"]
        cur_input_str = "<|im_start|>memory\nid:" + str(memory_id) + "\ntime:" + str(memory_time) + "\ncontent:" + str(
            memory_content) + "<|im_end|>"
        whole_str += cur_input_str

    query = example['query']
    query_time = example['query_time']
    query_str = "<|im_start|>query\n" + "time:" + str(query_time) + "\ncontent:" + query + "<|im_end|>"
    whole_str += query_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    whole_str += response
    # print("whole_str:", whole_str)
    related_memories = example['related_memories']
    low_related_memories = example['low_related_memories']
    print("query:", query)
    print("related:", related_memories, low_related_memories, response)
    type = example['type']
    if type not in total_nums:
        total_nums[type] = 1
    else:
        total_nums[type] += 1
    right = check_right(related_memories, low_related_memories, response)
    print("here:", right)
    if right:
        if type not in right_nums:
            right_nums[type] = 1
        else:
            right_nums[type] += 1

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Query SFT Training Script")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--train_json_path", type=str, required=True,
                        help="Path to the training JSON file")
    parser.add_argument("--test_json_path", type=str, required=True,
                        help="Path to the test JSON file")

    parser.add_argument("--output_dir", type=str, default="./Output/Memory_Query",
                        help="Model output directory")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_dir = args.model_dir
    train_json_path = args.train_json_path
    test_json_path = args.test_json_path
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    logging_steps = args.logging_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用8bit量化
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    
    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )

    train_df = pd.read_json(train_json_path)
    train_dataset = Dataset.from_pandas(train_df)
    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, num_proc=1)
    train_dataset = train_dataset.filter(filter_by_length, num_proc=1)
    train_dataset = train_dataset.shuffle()

    test_df = pd.read_json(test_json_path)
    test_dataset = Dataset.from_pandas(test_df)

    config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",

            ],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1,  # Dropout 比例
        )
    model = get_peft_model(model, config)
    print("load OK")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("begin train")
    trainer.train()

    model.eval()
    print("begin test")
    total_nums = {}
    right_nums = {}
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        predict(example, model, tokenizer, total_nums, right_nums)
    print("test end")
    print(total_nums)
    print(right_nums)