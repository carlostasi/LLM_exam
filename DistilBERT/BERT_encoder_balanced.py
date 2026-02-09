import torch
import tiktoken
import numpy as np
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datapreparation import load_and_prepare_data
from sklearn.metrics import accuracy_score
from datasets import concatenate_datasets, Dataset
import random

TARGET_SAMPLES = 3000 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# Here we load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Loading data...")
train, val, test, label_list, label2id, id2label = load_and_prepare_data()

# applying dataset balancing
train = balance_dataset(train, target_count=TARGET_SAMPLES)
print(f"New Training size: {len(train)}")

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
    output_dir="distilbert-classifier",
    num_train_epochs=3,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    # Evaluation and store of every epoch
    eval_strategy="epoch",
    save_strategy="epoch",
    # Use of a low learning rate to don't destroy the pre-trained knowledge of BERT
    learning_rate=2e-5,
    # Here we have the reload of the weights of the model that had the best accuracy
    load_best_model_at_end=True,
    metric_for_best_model="accuracy" # We could choose also "eval_loss"
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    compute_metrics = compute_metrics
)

print("\nStarting training...")
time_start = time.time()
trainer.train()
train_time = time.time() - time_start
print(f"\nTraining completed in {train_time:.2f} seconds ({train_time / 60:.2f}) minutes")

print("\n ==== EVALUATION METRICS ====")
# Evaluation on test dataset (data that the model had never seen before)
test_results = trainer.evaluate(tokenized_test)
print(f"Accuracy on test dataset: {test_results['eval_accuracy']}")

trainer.save_model("my_final_bert_classifier")
print("Model saved in 'my_final_bert_classifier'")

