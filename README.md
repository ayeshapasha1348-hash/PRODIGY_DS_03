# PRODIGY_DS_03
Decision Tree classifier on Bank Marketing dataset — predict customer subscription (Task-03).

# Task-03 — Decision Tree Classifier (Bank Marketing)

This project implements a Decision Tree classifier to predict whether a customer will subscribe to a product/service using the Bank Marketing dataset.

## Contents
- `notebook/decision_tree_bank_marketing.ipynb` — Jupyter notebook with step-by-step code and results.
- `src/train_model.py` — script to train the model and save the trained model file.
- `data/` — place dataset CSV here (`bank-marketing.csv`).
- `outputs/` — model, plots, and evaluation reports.

## How to run
1. Put `bank-marketing.csv` inside the `data/` folder.
2. From repo root:
   - To run notebook: open `notebook/decision_tree_bank_marketing.ipynb`
   - To train via script:
     ```
     python src/train_model.py --data data/bank-marketing.csv --out outputs/
     ```
3. Check `outputs/` for trained model (`.pkl`) and plots.

## Key points
- Tools: Python, pandas, scikit-learn, matplotlib
- Evaluation: accuracy, precision, recall, confusion matrix
- Outcome: interpretable decision rules and feature importance

