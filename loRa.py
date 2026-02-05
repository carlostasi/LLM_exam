import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import time
from datapreparation import load_and_prepare_data
import datapreparation # Your provided file

# 1. Load Data using the provided function [cite: 47]
train_ds, val_ds, test_ds, label_list, label2id, id2label = datapreparation.load_and_prepare_data()

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#by default is not defined
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)
model = get_peft_model(model,peft_config)

def prompt_format(subject,body,label=None):
    clean_body = body[:800]
    prompt = f"""
Classify the  following email into one of this department: 
Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.
Subject:{subject}
body:{body}
Department:""" 
    if label is not None:
        prompt += f" {label}{tokenizer.eos_token}"
    return prompt

def preprocess_function(examples):
    prompts = []
    for subj, body, queue in zip(examples['subject'], examples['body'], examples['queue']):
        prompts.append(prompt_format(subj,body,queue))
    model_input = tokenizer(prompts,max_length=512,truncation=True,padding="max_length")
    model_input["labels"] = model_input["input_ids"].copy()
    return model_input
print(f"preprocessing datasets ")
tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
print(f"training size : {len(tokenized_train)}")
print(f"validation size : {len(tokenized_val)}")