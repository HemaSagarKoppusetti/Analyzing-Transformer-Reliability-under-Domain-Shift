import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import time
import datetime
import random
import os
import sys

# Add the current directory to sys.path to find local modules
sys.path.insert(0, os.path.abspath('.'))

from data.load_dataset import load_imdb_dataset

# Set the seed for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 3
LEARNING_RATE = 2e-5

MODEL_NAME = "bert-base-uncased"

SAVE_PATH = "results/bert_imdb.pt"

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def train():

    train_loader, test_loader = load_imdb_dataset()

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        progress_bar = tqdm(train_loader)

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1} Loss {loss.item():.4f}" # Fixed: Added closing double quote and parenthesis
            )

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), SAVE_PATH)

    print("\nModel saved to", SAVE_PATH)


if __name__ == "__main__":

    train()
