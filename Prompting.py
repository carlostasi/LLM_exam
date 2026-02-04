import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datapreparation import load_and_prepare_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
print("Dati caricati!")
model_name = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#set the tokenaizer of the model to the end of the sentence 
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
def make_prompt(email_text):
    Prompt = f"""
Instruction: classify the following email into one of these categories: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry.
Email : {email_text}
Department:"""
    return Prompt
def predict(text):
    #create the prompt
    prompt = make_prompt(text)
    #tokenize it
    input = tokenizer(prompt, return_tensors="pt").to(device)
    #do_sample set to false to avoid to not pick the most likey word
    with torch.no_grad(): # avoid waste of ram
        output = model.generate(**input, max_new_tokens = 5,pad_token_id=tokenizer.eos_token_id,do_sample=False)
    answer = tokenizer.decode(output[0])
    if "Department:" in answer:
        answer = answer.split("Department:")[-1].strip()
    else:
        answer = "" 
        
    return answer

correct_answer = 0
for i in range(len(test_ds)):
    email = test_ds[i]['body']
    true_label = test_ds[i]['queue']

    prediction = predict(email)
    pred_clean = prediction.lower().strip()
    true_clean = true_label.lower().strip()

    if true_clean in pred_clean:
        correct_answer += 1
accuracy = (correct_answer / len(test_ds)) * 100
print("\n" + "="*30)
print(f"RISULTATO FINALE: {correct_answer}/{len(test_ds)} corretti")
print(f"Accuratezza: {accuracy:.2f}%")
print("="*30)