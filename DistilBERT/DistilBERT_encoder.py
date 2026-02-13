import torch
import tiktoken
import numpy as np
import time
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
from datapreparation import load_and_prepare_data
from sklearn.metrics import accuracy_score

def tokenize_function(elem):
    """
    This function takes an input 'elem', then creates a list of text pairs by combining 
    the elements of elem["subject"] and elem["body"] as strings.
    Then it uses the 'tokenizer' to tokenize the text pairs.
    Returns: the result of the tokenizer descpription
    """
    text_pairs = [str(s) + " " + str(b) for s, b in zip(elem["subject"], elem["body"])]

    return tokenizer(
        text_pairs,
        padding = "max_length",
        truncation = True,
        max_length = 512 # BERT standard constraint
    )

def map_labels(elem):
    """
    >>>> Mapping string labels --> int
    "Technical Support" == 0
    "Customer Service" == 1
    "Billing and Payments" == 2
    "Sales and Pre-Sales" == 3
    "General Inquiry" == 4
    """
    elem["labels"] = label2id[elem["queue"]]
    return elem

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}
    
device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "./modello_finale_bert"
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)
set_seed(SEED_VALUE)
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

# --- TRAINING CONFIGURATION ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    # Evaluation and store of every epoch
    eval_strategy="epoch",
    save_strategy="epoch",
    # Use of a low learning rate to don't destroy the pre-trained knowledge of BERT
    learning_rate=2e-5,
    # Here we have the reload of the weights of the model that had the best accuracy
    load_best_model_at_end=True,
    seed=SEED_VALUE,
    metric_for_best_model="accuracy", # We could choose also "eval_loss"
    save_total_limit=1
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    compute_metrics = compute_metrics
)

print("\nStarting training...")

# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()
time_start = time.time()
trainer.train()
train_time = time.time() - time_start
# memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"\nTraining completed in {train_time:.2f} seconds ({train_time / 60:.2f}) minutes")
# print(f"Memory used: {memory_gb:.3f} GB")

print("\n ==== EVALUATION METRICS ====")
# Evaluation on test dataset (data that the model had never seen before)
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()
test_results = trainer.evaluate(tokenized_test)
# memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Accuracy on test dataset: {test_results['eval_accuracy']}")
# print(f"Memory used for evaluation: {memory_gb:.3f} GB")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


