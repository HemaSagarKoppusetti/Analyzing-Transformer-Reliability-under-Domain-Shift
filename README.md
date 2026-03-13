
# Transformer Reliability Analysis under Domain Shift

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-IMDB%20%7C%20SST2-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

This project analyzes the **reliability of transformer-based NLP models under domain shift**.

A **BERT sentiment classifier** is trained on the **IMDB movie review dataset** and evaluated on both **IMDB** and **SST-2** datasets to analyze:

- cross-domain generalization
- confidence calibration
- prediction reliability
- model failure patterns

---

# Table of Contents

- Overview
- Research Objective
- Methodology
- Experimental Setup
- Results
- Calibration Analysis
- Error Analysis
- Repository Structure
- Installation
- Usage
- Future Work
- References

---

# Overview

Transformer models such as **BERT** achieve state-of-the-art results in many NLP tasks.  
However, when deployed in real-world applications, models often encounter **domain shift**, where the distribution of test data differs from the training dataset.

Understanding model behavior under such conditions is critical for building reliable machine learning systems.

---

# Research Objective

The goals of this project are:

- Evaluate transformer performance under domain shift
- Analyze prediction confidence and calibration
- Identify patterns in misclassified samples
- Understand robustness of pretrained language models

---

# Methodology

Model used:

```
bert-base-uncased
```

## Model Architecture

The sentiment classification system is built on top of the **BERT transformer architecture**.  
The contextual embedding of the `[CLS]` token is used as the representation for classification.

```mermaid
flowchart LR

A[Input Text] --> B[Tokenizer<br/>BERT Tokenization]

B --> C[BERT Transformer Encoder<br/>12 Layers Self-Attention]

C --> D[CLS Token Representation]

D --> E[Linear Classification Layer]

E --> F[Sentiment Prediction<br/>Positive / Negative]
```

Training objective: Cross-entropy loss for binary sentiment classification.

---

# Experimental Setup

## Experiment Workflow

The experimental pipeline evaluates the robustness of transformer models under **domain shift conditions**.

```mermaid
flowchart LR

A[IMDB Dataset] --> B[Fine-tune BERT Model]

B --> C[Trained Model Checkpoint]

C --> D[Evaluation on IMDB Test Set]

C --> E[Evaluation on SST-2 Dataset<br/>Domain Shift]

D --> F[Performance Metrics<br/>Accuracy, F1 Score]

E --> F

F --> G[Confidence Calibration Analysis]

F --> H[Error Analysis<br/>Misclassified Samples]
```

## Datasets

Two datasets were used.

### IMDB
- 50,000 movie reviews
- Balanced sentiment classes
- Long-form review text

### SST-2
- Stanford Sentiment Treebank
- Short movie review sentences
- Different linguistic distribution

Training domain: **IMDB**

Evaluation domains:
- IMDB (in-domain)
- SST-2 (domain shift)

---

# Training Configuration

| Parameter | Value |
|------|------|
Model | BERT-base-uncased |
Epochs | 3 |
Batch Size | 16 |
Learning Rate | 2e-5 |
Optimizer | AdamW |
Max Sequence Length | 128 |

---
## Evaluation Pipeline

```mermaid
flowchart TB

A[Model Predictions] --> B[Confusion Matrix]

A --> C[Confidence Distribution]

A --> D[Reliability Diagram]

A --> E[Error Analysis]

B --> F[Performance Evaluation]

C --> G[Calibration Analysis]

D --> G
```

# Results

## Classification Performance

| Dataset | Accuracy | F1 Score |
|------|------|------|
IMDB | **0.8826** | **0.8843** |
SST-2 | **0.8589** | **0.8701** |

Observation: Model performance decreases slightly under domain shift.

---

# Confusion Matrix

## IMDB

![IMDB Confusion Matrix](results/imdb_confusion_matrix.png)

## SST-2

![SST2 Confusion Matrix](results/sst2_confusion_matrix.png)

---

# Confidence Calibration Analysis

## IMDB Confidence Distribution

![IMDB Confidence Histogram](results/imdb_confidence_hist.png)

## IMDB Reliability Diagram

![IMDB Reliability Diagram](results/imdb_reliability.png)

## SST-2 Confidence Distribution

![SST2 Confidence Histogram](results/sst2_confidence_hist.png)

## SST-2 Reliability Diagram

![SST2 Reliability Diagram](results/sst2_reliability.png)

---

# Error Analysis

Misclassified examples were analyzed to identify common failure patterns.

Typical errors include:
- Short sentences with limited context
- Mixed sentiment statements
- Sarcastic expressions

Example:  
"The acting was great but the story was slow."

---

# Repository Structure

```
transformer-domain-shift
│
├── data
│   └── load_dataset.py
│
├── train
│   └── train_bert.py
│
├── eval
│   └── evaluate_model.py
│
├── experiments
│   ├── confidence_analysis.py
│   └── error_analysis.py
│
├── results
│
├── README.md
├── requirements.txt
└── environment.yml
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/transformer-domain-shift.git
cd transformer-domain-shift
```

Create environment:

```
conda create -n bert_reliability python=3.10
conda activate bert_reliability
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Usage

### Train Model

```
python train/train_bert.py
```

### Evaluate Model

```
python eval/evaluate_model.py
```

### Confidence Analysis

```
python experiments/confidence_analysis.py
```

### Error Analysis

```
python experiments/error_analysis.py
```
## Running with Docker

Build the Docker image:

```
docker build -t transformer-domain-shift
```

Run the container:

```
docker run -it --gpus all transformer-domain-shift
```

This will create an isolated environment containing all dependencies required to reproduce the experiments.
---

# Future Work

Possible research extensions include:

- Domain adaptation techniques
- Calibration-aware training
- Testing larger models (RoBERTa, DeBERTa)
- Cross-domain transfer learning experiments

---

# References

Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.

Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.

Pan & Yang, *A Survey on Transfer Learning*, IEEE TKDE 2010.

Hendrycks & Dietterich, *Benchmarking Neural Network Robustness*, ICLR 2019.

---

# Author

**Hema Sagar Koppusetti**  
B.Tech Artificial Intelligence and Machine Learning  
Lendi Institute of Engineering and Technology  

GitHub: https://github.com/HemaSagarKoppusetti  
LinkedIn: https://linkedin.com/in/hema-sagar-koppusetti
