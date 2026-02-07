import torch
from transformers import (AutoTokenizer,  AutoModelForCausalLM, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import time
from datapreparation import load_and_prepare_data
import datapreparation 
from datasets import concatenate_datasets


# 1. Load Data using the provided function 
train_ds, val_ds, test_ds, label_list, label2id, id2label = datapreparation.load_and_prepare_data()
print("Original Training size:", len(train_ds))
#balanced dataset
def get_balanced_dataset(input_ds, num_samples=3000):
    tech_support = input_ds.filter(lambda x: x['queue'] == 'Technical Support')
    others = input_ds.filter(lambda x: x['queue'] != 'Technical Support')
    tech_support = tech_support.shuffle(seed=42).select(range(num_samples))
    balanced_train_ds = concatenate_datasets([tech_support, others])
    balanced_train_ds = balanced_train_ds.shuffle(seed=42)
    return balanced_train_ds

# Call the function correctly
train_ds = get_balanced_dataset(train_ds, num_samples=3000)
# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "distilgpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# Carica modello base
base_model = AutoModelForCausalLM.from_pretrained(model_name)
# Carica i tuoi adattatori LoRA sopra il modello base

model.resize_token_embeddings(len(tokenizer))

tokenizer.padding_side = "right"


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn","c_proj", "c_fc"]
)
model = get_peft_model(model,peft_config)

def prompt_format(subject,body,label=None):
    clean_body = body[:800]
    prompt = f"""Classify each email into exactly ONE of these departments:
- Technical Support
- Customer Service
- Billing and Payments
- Sales and Pre-Sales
- General Inquiry
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
    input_ids = model_input["input_ids"]
    labels = []
    for seq in input_ids:
        seq_label = [token for token in seq]
        for i, token in enumerate(seq_label):
            if token == tokenizer.pad_token_id: 
                seq_label[i] = -100
        labels.append(seq_label)
    model_input["labels"] = labels
    return model_input
print(f"preprocessing datasets ")
tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
print(f"training size : {len(tokenized_train)}")
print(f"validation size : {len(tokenized_val)}")

training_args = TrainingArguments(
    output_dir="./lora_email_classifier",
    num_train_epochs=5,              
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
    marker = "Department:"
    if marker.lower() in generated.lower():
        answer = generated.lower().split(marker.lower())[-1].strip()
    else:
        answer = ""
    return answer

def departments_matching(prediction, valid_labels):
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
    pred_label = departments_matching(pred_text, label_list)
    
    if pred_text and pred_label == sample['queue']:
        correct += 1
    
    predictions.append(pred_label)
    true_labels.append(sample['queue'])

accuracy = (correct / total) * 100

print("\n" + "="*60)
print(f"MODEL: GPT2 + LoRA ")
print("="*60)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Training time: {train_time:.2f}s ({train_time/60:.1f} min)")
print("="*60)