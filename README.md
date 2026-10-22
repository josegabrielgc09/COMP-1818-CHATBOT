# COMP1818 - Intent Classification and Out-of-Scope Prediction

This repository contains the practical implementation for the COMP1818 Artificial Intelligence Applications coursework. The project investigates a Natural Language Understanding (NLU) pipeline capable of joint intent classification and out-of-scope (OOS) prediction.

## Project Overview
Communication breakdowns caused by out-of-scope queries severely degrade user satisfaction in conversational agents. This project compares two approaches:
* **Baseline Model:** Support Vector Machine (SVM) using TF-IDF feature extraction.
* **Proposed Model:** Fine-tuned DistilBERT (`distilbert-base-uncased`) for sequence classification.

## Dataset
The experiments use the **CLINC150 dataset**, which contains 150 in-scope intents and a dedicated subset of out-of-scope (OOS) queries. The dataset is automatically downloaded via the Hugging Face `datasets` library during execution.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
