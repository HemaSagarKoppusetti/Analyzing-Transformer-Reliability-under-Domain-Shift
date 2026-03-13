import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score
import sys
import os

# Add the current directory to sys.path to find local modules
sys.path.insert(0, os.path.abspath('.'))

from data.load_dataset import load_imdb_dataset, load_sst2_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/bert_imdb.pt"
MODEL_NAME = "bert-base-uncased"

BINS = 10


def get_predictions(model, dataloader):

    model.eval()

    all_probs = []
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

            probs = torch.softmax(outputs.logits, dim=1)

            conf, preds = torch.max(probs, dim=1)

            all_probs.extend(conf.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_ece(confidences, preds, labels):

    bin_boundaries = np.linspace(0, 1, BINS + 1)

    ece = 0

    for i in range(BINS):

        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:

            accuracy = np.mean(preds[in_bin] == labels[in_bin])
            avg_conf = np.mean(confidences[in_bin])

            ece += np.abs(avg_conf - accuracy) * prop_in_bin

    return ece


def plot_confidence_hist(confidences, title, save_path):

    plt.figure()

    plt.hist(confidences, bins=10)

    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.close()


def reliability_diagram(confidences, preds, labels, title, save_path):

    bin_boundaries = np.linspace(0, 1, BINS + 1)

    accs = []
    confs = []

    for i in range(BINS):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        mask = (confidences > lower) & (confidences <= upper)

        if np.sum(mask) > 0:

            acc = np.mean(preds[mask] == labels[mask])
            conf = np.mean(confidences[mask])

        else:

            acc = 0
            conf = 0

        accs.append(acc)
        confs.append(conf)

    plt.figure()

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.plot(confs, accs, marker="o", label="Model")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.legend()

    plt.savefig(save_path)
    plt.close()


def main():

    print("Loading datasets...")

    imdb_train, imdb_test = load_imdb_dataset()
    sst2_test = load_sst2_dataset()

    print("Loading model...")

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

    print("Running IMDB confidence analysis...")

    imdb_conf, imdb_preds, imdb_labels = get_predictions(model, imdb_test)

    imdb_ece = compute_ece(imdb_conf, imdb_preds, imdb_labels)

    print("IMDB Accuracy:", accuracy_score(imdb_labels, imdb_preds))
    print("IMDB ECE:", imdb_ece)

    os.makedirs("results", exist_ok=True)
    plot_confidence_hist(
        imdb_conf,
        "Confidence Distribution (IMDB)",
        "results/imdb_confidence_hist.png"
    )

    reliability_diagram(
        imdb_conf,
        imdb_preds,
        imdb_labels,
        "Reliability Diagram (IMDB)",
        "results/imdb_reliability.png"
    )

    print("Running SST-2 confidence analysis...")

    sst_conf, sst_preds, sst_labels = get_predictions(model, sst2_test)

    sst_ece = compute_ece(sst_conf, sst_preds, sst_labels)

    print("SST2 Accuracy:", accuracy_score(sst_labels, sst_preds))
    print("SST2 ECE:", sst_ece)

    plot_confidence_hist(
        sst_conf,
        "Confidence Distribution (SST2)",
        "results/sst2_confidence_hist.png"
    )

    reliability_diagram(
        sst_conf,
        sst_preds,
        sst_labels,
        "Reliability Diagram (SST2)",
        "results/sst2_reliability.png"
    )


if __name__ == "__main__":
    main()
