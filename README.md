# COMP1818 - Intent Classification and Out-of-Scope Prediction

This repository contains the practical implementation for the COMP1818 Artificial Intelligence Applications coursework. The project investigates a Natural Language Understanding (NLU) pipeline capable of joint intent classification and out-of-scope (OOS) prediction.

## Project Overview
Communication breakdowns caused by out-of-scope queries severely degrade user satisfaction in conversational agents. This project compares two approaches:
* **Baseline Model:** Support Vector Machine (SVM) using TF-IDF feature extraction.
* **Proposed Model:** Fine-tuned DistilBERT (`distilbert-base-uncased`) for sequence classification.

## Dataset
The experiments use the **CLINC150 dataset**, which contains 150 in-scope intents and a dedicated subset of out-of-scope (OOS) queries. The dataset is automatically downloaded via the Hugging Face `datasets` library during execution.

## Requirements
The code is designed to run in a Jupyter/Colab environment. The required libraries are:
`transformers`, `datasets`, `scikit-learn`, `torch`, `numpy`

## How to Run
1. Open the `main.ipynb` notebook.
2. The notebook is fully self-contained. Run the first cell to install the dependencies.
3. Run the second cell to execute the training and evaluation pipeline for both the SVM baseline and the DistilBERT model.
4. The output will display the Accuracy metrics for both models. For optimal training speed with the Transformer model, it is recommended to run this notebook in Google Colab with a GPU runtime enabled.
