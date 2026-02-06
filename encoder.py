import torch
import tiktoken
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datapreparation import load_and_prepare_data
from sklearn.metrics import accuracy_score


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(elem):
    text_pairs = [s + " " + b for s, b in zip(elem["subject"], elem["body"])]

    return tokenizer(
        text_pairs,
        padding = "max_length",
        truncation = True,
        max_length = 512 # BERT standard constraint
    )

def map_labels(elem):
    elem["labels"] = label2id[elem["queue"]]
    return elem

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}
    

model = AutoModelForSequenceClassification.from_pretrained (
    "distilbert-base-uncased",
    num_labels=5
)
model.to(device)

train, val, test, label_list, label2id, id2label = load_and_prepare_data()

train = train.map(map_labels)
val = val.map(map_labels)
test = test.map(map_labels)

tokenized_train = train.map(tokenize_function, batched=True)
tokenized_val = val.map(tokenize_function, batched=True)
tokenized_test = test.map(tokenize_function, batched=True)

args = TrainingArguments(
    output_dir="distilbert-classifier",
    num_train_epochs=3,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    compute_metrics = compute_metrics
)