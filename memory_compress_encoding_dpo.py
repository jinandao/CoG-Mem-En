import pandas as pd
from datasets import Dataset
from functools import partial
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import torch
import random
import argparse

def preprocess_dpo_data(example, tokenizer):
    prompt_input_str = f"<|im_start|>system\nYou are an AI assistant skilled at summarizing and condensing conversations. You need to extract and sort out the key points of the conversation in the <think></think> blocks, and then write a final summary in the <memory></memory> section."
    conversations = example['conversation']
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
        prompt_input_str += cur_input_str
    prompt_input_str_ids = tokenizer(prompt_input_str, add_special_tokens=False)
    chosen_part = example['chosen']
    chosen_think_str = chosen_part['think']
    chosen_memory_str = chosen_part['memory']
    ret_chosen_label_str = "\nsummary:<think>" + chosen_think_str + "</think><memory>" + chosen_memory_str + "</memory>" + tokenizer.eos_token
    # chosen_tokens = tokenizer(chosen_label_str, add_special_tokens=False)
    rejected_part = example['rejected']
    rejected_think_str = rejected_part['think']
    rejected_memory_str = rejected_part['memory']
    ret_rejected_label_str = "\nsummary:<think>" + rejected_think_str + "</think><memory>" + rejected_memory_str + "</memory>" + tokenizer.eos_token
    # rejected_tokens = tokenizer(rejected_label_str, add_special_tokens=False)
    return {
        'prompt': prompt_input_str,
        'chosen': ret_chosen_label_str,
        'rejected': ret_rejected_label_str,
    }

def predict(example, model, tokenizer):
    # assert False
    conversations = example['conversation']
    whole_str = f"<|im_start|>system\nYou are an AI assistant skilled at summarizing and condensing conversations. You need to extract and sort out the key points of the conversation in the <think></think> blocks, and then write a final summary in the <memory></memory> section."
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
        else:
            cur_input_str = "<|im_start|>assistant\n" +  conversations[i]['content'] + "<|im_end|>\n"
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
    parser = argparse.ArgumentParser(description="Memory Compression DPO Training Script")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model")

    parser.add_argument("--train_json_path", type=str, required=True,
                        help="Path to the training set JSON file")

    parser.add_argument("--sft_lora_path", type=str, required=True,
                        help="Path to the SFT-trained LoRA model")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the DPO model")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=2560,
                        help="Maximum prompt length")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 mixed precision training")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                        help="Maximum gradient norm")
    parser.add_argument("--remove_unused_columns", action="store_true",
                        help="Whether to remove unused columns")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed")
    parser.add_argument("--continue_train", action="store_true",
                        help="Whether to continue training from the SFT model")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model_dir = args.model_dir
    train_json_path = args.train_json_path
    sft_lora_path = args.sft_lora_path
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    output_dir = args.output_dir
    beta = args.beta
    max_length = args.max_length
    max_prompt_length = args.max_prompt_length
    gradient_checkpointing = args.gradient_checkpointing
    fp16 = args.fp16
    max_grad_norm = args.max_grad_norm
    remove_unused_columns = args.remove_unused_columns
    num_train_epochs = args.num_train_epochs
    seed = args.seed
    continue_train = args.continue_train

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

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )

    if continue_train:
        lora_model_path = sft_lora_path
        model = PeftModel.from_pretrained(model, lora_model_path, is_trainable=True)
        ref_model = PeftModel.from_pretrained(ref_model, lora_model_path, is_trainable=False)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    if not continue_train:
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

    preprocess_dpo_data = partial(preprocess_dpo_data, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_dpo_data, remove_columns=dataset.column_names)
    print(train_dataset)

    args = DPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  
        learning_rate=learning_rate,
        optim='rmsprop',
        report_to='none',
        output_dir=output_dir,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,  
        max_grad_norm=max_grad_norm,  
        remove_unused_columns=remove_unused_columns,  # 重要：不要自动移除列
        num_train_epochs=num_train_epochs,
    )

    dpo_trainer = DPOTrainer(model,
                         ref_model,
                         args=args,
                         train_dataset=train_dataset,
                         processing_class=tokenizer)

    print("start DPO training...")
    dpo_trainer.train()

    model.eval()
    print("-----------begin test-----------")
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        predict(example, model, tokenizer)
    print("-----------end test-----------")