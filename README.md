# COMP1818 - Intent Classification and Out-of-Scope Prediction

This repository contains the practical implementation for the **COMP1818 Artificial Intelligence Applications** coursework (University of Greenwich). The project investigates a Natural Language Understanding (NLU) pipeline capable of joint intent classification and out-of-scope (OOS) prediction.

## Project Overview

Communication breakdowns caused by out-of-scope queries severely degrade user satisfaction in conversational agents. This project compares two approaches:
* **Baseline Model**: Support Vector Machine (SVM) using TF-IDF feature extraction.
* **Proposed Model**: Fine-tuned DistilBERT (`distilbert-base-uncased`) for sequence classification.

## Dataset

The experiments use the **CLINC150 dataset** (Larson et al., 2019), which contains 150 in-scope intents and a dedicated subset of out-of-scope (OOS) queries. The dataset is automatically downloaded via the Hugging Face `datasets` library during execution. No manual downloading is required.

## Requirements

The codebase is structured as a self-contained Jupyter Notebook, optimized for execution in Google Colab (with GPU acceleration for DistilBERT). The required dependencies are:
* `transformers`
* `datasets`
* `scikit-learn`
* `torch`
* `numpy`
* `pandas`

## How to Run the Code

To ensure an error-free evaluation pipeline as required by the coursework brief:
1. Open the `main.ipynb` file in **Google Colab**.
2. Go to `Runtime` -> `Change runtime type` and select **GPU** (T4 or higher).
3. Run all cells sequentially (`Runtime` -> `Run all`). 
4. The notebook will automatically:
   - Install the required dependencies.
   - Download the CLINC150 dataset.
   - Train and evaluate the TF-IDF + SVM baseline.
   - Fine-tune and evaluate the DistilBERT model.
   - Output the final Accuracy, Precision, Recall, and F1 metrics for comparison.

## Author
* **Jose Gabriel Gutierrez Claros** (ID: 001390237)
