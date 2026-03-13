from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LENGTH = 128
BATCH_SIZE = 16


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


def load_imdb_dataset():
    
    imdb = load_dataset("imdb")

    imdb = imdb.map(tokenize_function, batched=True)

    imdb.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(
        imdb["train"],
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        imdb["test"],
        batch_size=BATCH_SIZE
    )

    return train_loader, test_loader


def load_sst2_dataset():

    sst2 = load_dataset("glue", "sst2")

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    sst2 = sst2.map(preprocess, batched=True)

    sst2.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    test_loader = DataLoader(
        sst2["validation"],
        batch_size=BATCH_SIZE
    )

    return test_loader


if __name__ == "__main__":

    imdb_train, imdb_test = load_imdb_dataset()
    sst2_test = load_sst2_dataset()

    print("IMDB train batches:", len(imdb_train))
    print("IMDB test batches:", len(imdb_test))
    print("SST2 test batches:", len(sst2_test))