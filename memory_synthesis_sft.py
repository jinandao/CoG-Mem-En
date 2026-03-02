import json
from functools import partial

import pandas as pd
import torch
from datasets import Dataset
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
import argparse

def process_func(example, tokenizer):
    messages = example['conversations']
    input_str= f"<|im_start|>system\nYou are an AI assistant having a conversation with a user. When the user mentions things from the past, you need to call the memory_query_call function and pass the query as the parameter to retrieve memories. Then, based on the memories returned by the memory_query role, think through the information of the memory fragments within the blocks, and generate a proper response according to the organized information. If the user does not mention anything from the past, generate a proper response based on the current context."
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        else:
            cur_input_str = ""
            if 'think' in messages[i] and messages[i]['think'] is not None:
                cur_input_str += "<think>" + messages[i]['think'] + "</think>"
            cur_input_str += messages[i]['content'] + tokenizer.eos_token
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend(cur_input_ids['input_ids'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    return example["length"] <= 4096

def predict(example, model, tokenizer):
    messages = example['conversations']
    whole_str = f"<|im_start|>system\nYou are an AI assistant having a conversation with a user. When the user mentions things from the past, you need to call the memory_query_call function and pass the query as the parameter to retrieve memories. Then, based on the memories returned by the memory_query role, think through the information of the memory fragments within the blocks, and generate a proper response according to the organized information. If the user does not mention anything from the past, generate a proper response based on the current context."
    
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                    max_new_tokens=384,
                                    temperature=0.1,
                                    do_sample=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                     max_new_tokens=384,
                                     temperature=0.1,
                                     do_sample=True,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id, )
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
    print("whole_str:", whole_str)
    print("--------------------------")

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Conversation SFT Training Script")

    parser.add_argument("--train_json_path", type=str, required=True,
                        help="Path to the training JSON file")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pre-trained model")

    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./Output/Conversation_Use_Memory",
                        help="Model output directory")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument("--num_train_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_json_path = args.train_json_path
    model_dir = args.model_dir
    seed = args.seed
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    logging_steps = args.logging_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing

    train_df = pd.read_json(train_json_path)
    dataset = Dataset.from_pandas(train_df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed, shuffle=True)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16, 
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

    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names, num_proc=4)
    train_dataset = train_dataset.filter(filter_by_length, num_proc=4) 
    print(train_dataset) 
    print(test_dataset)

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
        inference_mode=False,  
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.1, 
    )
    model = get_peft_model(model, config)
    print("load OK")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=10,
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
    print("-----------begin test-----------")
    test_samples = min(50, len(test_dataset))
    for i in range(test_samples):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("-----------end test-----------")