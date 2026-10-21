# Customer Churn Prediction (Telco)

Analysis and modeling of customer churn for a telecom operator, based on the Kaggle [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (7,043 customers, 21 variables).

**Goal:** identify customers likely to cancel their subscription (churn) and understand what drives them to leave, in order to target retention actions.

## Key results

- **26.5%** of customers churned: the dataset is imbalanced.
- The selected model is **logistic regression**: its test AUC (0.839) is close to the Random Forest's (0.845), and it remains interpretable.
- At a 0.5 threshold, it detects **78% of the customers who leave** (recall), with 51% precision.
- Factors that increase churn risk: fiber optic internet, electronic check payment, paperless billing, streaming options.
- Factors that reduce it: tenure, one- or two-year contracts, online security and tech support options.

### Model comparison

5-fold stratified cross-validation on the training set, then evaluation on the test set (20% of the data).

| Model | AUC (cross-validation) | Recall (CV) | Precision (CV) | AUC (test) |
|---|---|---|---|---|
| Baseline (dummy) | 0.507 | 0.278 | 0.275 | 0.516 |
| Logistic regression | 0.845 | 0.794 | 0.521 | 0.839 |
| Random Forest | 0.846 | 0.759 | 0.542 | 0.845 |
| Gradient Boosting | 0.846 | 0.796 | 0.526 | 0.839 |

The three models perform very similarly, so the simplest one is kept in order to read its coefficients.

### Choosing the decision threshold

The threshold depends on the cost of a retention action compared with the cost of losing a customer.

| Threshold | Customers flagged | Recall | Precision |
|---|---|---|---|
| 0.3 | 805 | 93% | 43% |
| 0.4 | 687 | 86% | 47% |
| **0.5** | 571 | 78% | 51% |
| 0.6 | 476 | 70% | 55% |
| 0.7 | 362 | 60% | 62% |

### Risk factors (logistic regression odds ratios)

| Variable | Odds ratio |
|---|---|
| Fiber optic internet | 3.06 |
| Electronic check payment | 1.49 |
| Streaming movies / TV | 1.48 / 1.45 |
| Paperless billing | 1.40 |
| Tenure | 0.47 |
| 1-year / 2-year contract | 0.49 / 0.25 |

An odds ratio above 1 increases the odds of churn; an odds ratio below 1 reduces them.

## Method

1. **Preparation**: drop `customerID`; convert `TotalCharges` to numeric; merge "No internet service" and "No phone service" into "No"; drop `TotalCharges`, which is redundant with `tenure × MonthlyCharges`.
2. **Exploration**: churn distribution, churn rate per variable, distributions, correlations (figures 01 to 07).
3. **Modeling**: scikit-learn pipeline (scaling of numeric variables, one-hot encoding of categorical ones), stratified train/test split, `class_weight="balanced"` to compensate for the imbalance.
4. **Evaluation**: AUC, recall, precision, F1, confusion matrix, precision/recall curve by threshold (figures 08 to 11).

## Project structure

```
.
├── download_kaggle_data.py   # downloads the dataset into data/raw/
├── exploration.py            # exploratory analysis -> figures 01 to 07
├── modelisation.py           # model comparison -> figures 08 to 11
├── figures/                  # generated charts
└── data/raw/                 # data (not versioned)
```

## Installation and usage

Requirements: Python 3.10+.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kaggle
```

1. **Data**: create a token at [kaggle.com/settings](https://www.kaggle.com/settings), put `kaggle.json` in `~/.kaggle/`, then run:
   ```bash
   python download_kaggle_data.py
   ```
2. **Exploration**:
   ```bash
   python exploration.py
   ```
3. **Modeling**:
   ```bash
   python modelisation.py
   ```

Run the scripts from the project root. Figures are saved in `figures/`.

## Possible improvements

- Tune hyperparameters (grid search) and calibrate the probabilities.
- Compute the optimal threshold from the cost of a retention offer and the value of a customer.
- Try other models (XGBoost, LightGBM) and feature interactions.
