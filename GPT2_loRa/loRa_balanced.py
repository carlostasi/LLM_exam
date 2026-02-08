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
import random
import numpy as np
from datasets import concatenate_datasets, Dataset
import datapreparation 


TARGET_SAMPLES = 3000 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. loading data
train_ds, val_ds, test_ds, label_list, label2id, id2label = datapreparation.load_and_prepare_data()
print(f"Original Training size: {len(train_ds)}")

# manual data augmentation
def augment_text(text):
    words = text.split()
    #if the sentece is too small, leave it 
    if len(words) < 5:
        return text
    # creation of a copy 
    new_words = words[:]
    
    # choose two random indexes to change
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return " ".join(new_words)

def balance_dataset(dataset, target_count=3000):
    print("Balancing training data distribution...")
    
    # separating the dataset for deparment
    class_datasets = {}
    for label in label_list:
        class_datasets[label] = dataset.filter(lambda x: x['queue'] == label)
    
    final_parts = []
    
    for label, ds in class_datasets.items():
        count = len(ds)
        
        # add original data
        final_parts.append(ds)
        
        if count < target_count:
            missing = target_count - count
            indices = np.random.choice(count, missing, replace=True)
            new_data = {'subject': [], 'body': [], 'queue': [], 'language': []}
            
            for idx in indices:
                item = ds[int(idx)]
                # applying varietion to the body
                new_body = augment_text(item['body'])
                new_data['body'].append(new_body)
                new_data['subject'].append(item['subject']) 
                new_data['queue'].append(label)
                new_data['language'].append('en')
            
            # creating the dataset 
            augmented_ds = Dataset.from_dict(new_data)
            final_parts.append(augmented_ds)
        else:
            ds = ds.shuffle(seed=SEED).select(range(target_count))
            final_parts.pop() 
            final_parts.append(ds)
            
    return concatenate_datasets(final_parts).shuffle(seed=SEED)

# applying dataset balancing
train_ds = balance_dataset(train_ds, target_count=TARGET_SAMPLES)
print(f"New Training size: {len(train_ds)}")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "distilgpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configuration LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def prompt_format(subject, body, label=None):
    clean_body = str(body)[:800]
    prompt = f"""Classify customer emails into exactly one of the following departments.

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
    subjects = examples['subject']
    bodies = examples['body']
    queues = examples['queue']
    
    for subj, body, queue in zip(subjects, bodies, queues):
        prompts.append(prompt_format(subj, body, queue))
        
    model_input = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    model_input["labels"] = model_input["input_ids"].copy()
    return model_input

print(f"Tokenizing datasets...")
tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)

training_args = TrainingArguments(
    output_dir="./lora_email_classifier",
    num_train_epochs=3,              
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,    
    gradient_accumulation_steps=4,   
    learning_rate=2e-4,              
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
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

print(f"Starting training")
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds")

# model saving
model.save_pretrained("./lora_model_final")
tokenizer.save_pretrained("./lora_model_final")


def department_prediction(subject, body):
    model.eval()
    prompt = prompt_format(subject, body, None)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=False
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    marker = "Department:"
    if marker in generated:
        answer = generated.split(marker)[-1].strip()
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

correct = 0
total = len(test_ds)
print(f"Starting evaluation on {total} test samples...")

for i in tqdm(range(total), desc="Testing"):
    sample = test_ds[i]
    pred_text = department_prediction(sample['subject'], sample['body'])
    pred_label = departments_match(pred_text, label_list)
    
    if pred_label == sample['queue']:
        correct += 1

accuracy = (correct / total) * 100

print(f"MODEL: DistilGPT2 + LoRA (Balanced)")
print("="*60)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Training time: {train_time:.2f}s ({train_time/60:.1f} min)")