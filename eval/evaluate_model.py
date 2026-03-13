import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import sys
import os

# Add the current directory to sys.path to find local modules
sys.path.insert(0, os.path.abspath('.'))

from data.load_dataset import load_imdb_dataset, load_sst2_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/bert_imdb.pt"
MODEL_NAME = "bert-base-uncased"

def evaluate(model, dataloader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1, cm


def main():

    print("\nLoading datasets...")

    # load_imdb_dataset returns train_loader, test_loader
    _, imdb_test = load_imdb_dataset()
    sst2_test = load_sst2_dataset()

    print("Loading trained model...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the training script ran successfully and saved the model.")
        return # Exit if model not found

    model.to(DEVICE)

    print("\nEvaluating on IMDB (In-Domain)...")

    imdb_acc, imdb_f1, imdb_cm = evaluate(model, imdb_test)

    print("\nEvaluating on SST-2 (Domain Shift)...")

    sst2_acc, sst2_f1, sst2_cm = evaluate(model, sst2_test)

    print("\n===== RESULTS ===")

    print("\nIMDB Test Results")
    print("Accuracy:", imdb_acc)
    print("F1 Score:", imdb_f1)
    print("Confusion Matrix:\n", imdb_cm)

    print("\nSST-2 Test Results")
    print("Accuracy:", sst2_acc)
    print("F1 Score:", sst2_f1)
    print("Confusion Matrix:\n", sst2_cm)


if __name__ == "__main__":
    main()
