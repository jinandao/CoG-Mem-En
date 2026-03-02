import json
from functools import partial

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
import argparse


def process_func(example, tokenizer):
    conversations = example['conversation']
    input_str = f"<|im_start|>system\nYou are an AI assistant skilled at summarizing and condensing conversations. You need to extract and sort out the key points of the conversation in the <think></think> blocks, and then write a final summary in the <memory></memory> section."
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
        cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        labels.extend([-100] * len(cur_input_ids['input_ids']))
        input_str += cur_input_str
    think_str = example['think']
    memory_str = example['memory']
    label_str = "\nsummary:<think>" + think_str + "</think><memory>" + memory_str + "</memory>" + tokenizer.eos_token

    input_str += label_str
    cur_input_ids = tokenizer(label_str, add_special_tokens=False)
    input_ids.extend(cur_input_ids['input_ids'])
    attention_mask.extend(cur_input_ids['attention_mask'])
    labels.extend(cur_input_ids['input_ids'])

    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    return example["length"] <= 4096

def predict(example, model, tokenizer):
    conversations = example['conversation']
    whole_str = f"<|im_start|>system\nYou are an AI assistant skilled at summarizing and condensing conversations. You need to extract and sort out the key points of the conversation in the <think></think> blocks, and then write a final summary in the <memory></memory> section."
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n"+ conversations[i]['content'] + "<|im_end|>\n"
        whole_str += cur_input_str
    inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024,
                                    temperature=0.1,
                                    do_sample=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    whole_str += response
    print("whole_str:", whole_str)
    print("--------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Compression SFT Training Script")

    parser.add_argument("--train_json_path", type=str, required=True,
                        help="Path to the training JSON file")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pre-trained model")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the SFT model")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--seed", type=int, default=145,
                        help="Random seed")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_json_path = args.train_json_path
    model_dir = args.model_dir
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing
    seed = args.seed

    train_df = pd.read_json(train_json_path)
    dataset = Dataset.from_pandas(train_df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=seed, shuffle=True)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    process_func = partial(process_func, tokenizer=tokenizer)
    train_dataset = train_dataset.map(process_func, num_proc=1)
    # train_dataset = train_dataset.map(process_func_no_COT, num_proc=1)

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

    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)

    model.eval()
    print("-----------begin test-----------")
    test_samples = min(10, len(test_dataset))
    for i in range(test_samples):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("-----------end test-----------")