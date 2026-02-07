import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import time
from datapreparation import load_and_prepare_data
import datapreparation 
from datasets import concatenate_datasets


# 1. Load Data using the provided function [cite: 47]
train_ds, val_ds, test_ds, label_list, label2id, id2label = datapreparation.load_and_prepare_data()
print("Original Training size:", len(train_ds))

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
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)
model = get_peft_model(model,peft_config)

def prompt_format(subject,body,label=None):
    clean_body = body[:800]
    prompt = f"""Classify customer emails into exaclty one of the following departments.

Examples:
Email: Password reset not working
Department: Technical Support

Email: I want a refund for order #123
Department: Billing and Payments

Email: Tell me about your enterprise plan
Department: Sales and Pre-Sales

Email: How do I change my settings?
Department: Customer Service

Email: General question about your company
Department: General Inquiry

Now classify:
Email: Subject: {subject}
{clean_body}
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

training_args = TrainingArguments(
    output_dir="./lora_email_classifier",
    num_train_epochs=3,              
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   
    learning_rate=2e-4,              
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=(device == "cuda"),         
    report_to="none",                
    warmup_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)
print(f"starting the training session")
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds")
#save the model
model.save_pretrained("./lora_model_final")
tokenizer.save_pretrained("./lora_model_final")

def department_prediction(subject,body):
    prompt = prompt_format(subject,body,None)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs,max_new_tokens=15, pad_token_id=tokenizer.eos_token_id,do_sample=False)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "Best matching department:"
    if marker.lower() in generated.lower():
        answer = generated.lower().split(marker.lower())[-1].strip()
    else:
        answer = ""
    return answer

def departments_match(prediction, valid_labels):
    pred_lower = prediction.lower().strip()
    for label in valid_labels:
        if pred_lower == label.lower():
            return label
    for label in valid_labels:
        if pred_lower.startswith(label.lower()):
            return label
    sorted_labels = sorted(valid_labels, key=len, reverse=True)
    for label in sorted_labels:
        if label.lower() in pred_lower:
            return label
    return None

from tqdm import tqdm

correct = 0
total = len(test_ds)
predictions = []
true_labels = []

for i in tqdm(range(total), desc="Testing"):
    sample = test_ds[i]
    pred_text = department_prediction(sample['subject'], sample['body'])
    pred_label = departments_match(pred_text, label_list)
    
    if pred_text and pred_label == sample['queue']:
        correct += 1
    
    predictions.append(pred_label)
    true_labels.append(sample['queue'])

accuracy = (correct / total) * 100

print("\n" + "="*60)
print(f"MODEL: GPT2 + LoRA")
print("="*60)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Training time: {train_time:.2f}s ({train_time/60:.1f} min)")
print("="*60)

