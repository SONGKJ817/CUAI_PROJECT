import pandas as pd
import torch
import json
import os
from peft import LoraConfig
import bitsandbytes as bnb
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from peft import get_peft_model


# 메모리 절약을 목적으로 데이터를 저비트로 압축하는 라이브러리
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 가중치를 4비트로 압축
    bnb_4bit_quant_type="nf4", # negative floating-point-4-bit
    bnb_4bit_compute_dtype=torch.bfloat16, # 16비트 brain floating point
)

# utils
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


login(token='') # huggingface token을 입력하세요.


# PATH 설정
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
checkpoint = "final_checkpoint"
output_dir = "" # output directory를 입력하세요.

# load dataset
dataset = load_dataset("json", data_files={"train" : PATH + 'data/QA_Dataset.json'})
train_dataset = dataset["train"]
train_dataset

# model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map='auto')
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"""너는 질문에 대한 답변을 내놓는 인공지능 모델이야.
        질문 : ```{example['question'][i]}```\n
        답변: {example['answer'][i]}"""
        output_texts.append(text)
    return output_texts


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
from transformers import TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    bf16=True, # gpu 종류에 따라 바꾸기 (bf16, fp16)
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

from trl import SFTTrainer

trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=4096,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
torch.cuda.empty_cache()