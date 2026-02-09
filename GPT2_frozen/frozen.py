import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datapreparation import load_and_prepare_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
print("loaded data")
torch.cuda.empty_cache()  
torch.cuda.reset_peak_memory_stats()
#to change model write : "gpt2" 
model_name = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def make_prompt(subject, body):
    prompt = f"""Task: Classify each email into exactly ONE of these departments:
- Technical Support
- Customer Service
- Billing and Payments
- Sales and Pre-Sales
- General Inquiry

Examples:

Email: My internet connection keeps dropping. Can you help?
Department: Technical Support

Email: I want to purchase your premium plan. What are the options?
Department: Sales and Pre-Sales

Email: My credit card was charged twice for the same order
Department: Billing and Payments

Email: How do I reset my account password?
Department: Customer Service

Now classify this email:

Email: Subject: {subject}
{body}
Department:"""
    return prompt

def extract_department(answer, label_list):
    answer_lower = answer.lower().strip()
    for label in label_list:
        if label.lower() in answer_lower:
            return label
    return None

def predict(subject, body):
    prompt = make_prompt(subject, body)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    full_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the generated part after "Department:"
    if "Department:" in full_answer:
        answer = full_answer.split("Department:")[-1].strip()
    else:
        answer = ""
    
    return answer

# Evaluation
print("\nEvaluating on test set...")
start_time = time.time()
correct_answer = 0

for i in range(len(test_ds)):
    subject = test_ds[i]['subject']
    body = test_ds[i]['body']
    true_label = test_ds[i]['queue']
    
    prediction = predict(subject, body)
    pred_dept = extract_department(prediction, label_list)
    
    if pred_dept and pred_dept.lower() == true_label.lower():
        correct_answer += 1

end_time = time.time()

accuracy = (correct_answer / len(test_ds)) * 100
print("\n" + "="*50)
print(f"Model: {model_name} (frozen, prompting)")
print(f"Correct: {correct_answer}/{len(test_ds)}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Time: {end_time - start_time:.2f}s")
print("="*50)
memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) 
print(f"Memory used: {memory_gb:.3f} GB")