import torch
import tiktoken
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datapreparation import load_and_prepare_data
from sklearn.metrics import accuracy_score

def tokenize_function(elem):
    text_pairs = [str(s) + " " + str(b) for s, b in zip(elem["subject"], elem["body"])]

    return tokenizer(
        text_pairs,
        padding = "max_length",
        truncation = True,
        max_length = 512 # BERT standard constraint
    )
"""
 >>>> Mapping string labels --> int
"Technical Support" == 0
"Customer Service" == 1
"Billing and Payments" == 2
"Sales and Pre-Sales" == 3
"General Inquiry" == 4
"""
def map_labels(elem):
    
    elem["labels"] = label2id[elem["queue"]]
    return elem

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}
    

device = "cuda" if torch.cuda.is_available() else "cpu"

# Here we load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Loading data...")
train, val, test, label_list, label2id, id2label = load_and_prepare_data()

print("Dataset mapping...")
train = train.map(map_labels)
val = val.map(map_labels)
test = test.map(map_labels)

print("\nTokenize...")
# Text --> Tensor
tokenized_train = train.map(tokenize_function, batched=True)
tokenized_val = val.map(tokenize_function, batched=True)
tokenized_test = test.map(tokenize_function, batched=True)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained (
    "distilbert-base-uncased",
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# --- TRAINING PART ---
args = TrainingArguments(
    output_dir="distilbert-classifier",
    num_train_epochs=1,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    compute_metrics = compute_metrics
)

print("\nStarting training...")
trainer.train()

print("\n ==== EVALUTAION METRICS ====")
test_results = trainer.evaluate(tokenized_test)
print(f"Accuracy: {test_results['eval_accuracy']}")

trainer.save_model("my_final_bert_classifier")
print("Modello salvato in 'my_final_bert_classifier'")

