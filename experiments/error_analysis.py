import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "results/bert_imdb.pt"

MAX_LENGTH = 128


def load_model():

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    return model


def predict(model, tokenizer, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()


def run_error_analysis():

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = load_model()

    sst2 = load_dataset("glue", "sst2")

    validation = sst2["validation"]

    errors = []

    for sample in validation:

        text = sample["sentence"]
        label = sample["label"]

        pred, conf = predict(model, tokenizer, text)

        if pred != label:

            errors.append({
                "text": text,
                "true_label": label,
                "predicted_label": pred,
                "confidence": conf
            })

    df = pd.DataFrame(errors)

    print("Total errors:", len(df))

    df.to_csv("results/misclassified_examples.csv", index=False)

    print("Saved misclassified examples to results/misclassified_examples.csv")


if __name__ == "__main__":

    run_error_analysis()