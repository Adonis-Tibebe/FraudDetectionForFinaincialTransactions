# Scripts

This directory contains the main training scripts for the fraud detection project. These scripts automate the end-to-end process of data loading, preprocessing, model training, evaluation, and artifact saving for both the fraud and credit card datasets.

---

## Contents

- **Train_Fraud_model.py**
- **Train_CreditCard_model.py**

---

## Script Details

### `Train_Fraud_model.py`

- **Purpose:**  
  Trains and evaluates models on the feature-engineered fraud dataset.

- **Workflow:**
  1. Loads the feature-engineered fraud data from `../data/processed/Feature_engineered/Feature_engineered_fraud_data.csv`.
  2. Initializes the `FraudPreprocessor` in `fraud_data` mode, which:
     - Applies frequency encoding to `device_id` and target encoding to `country`.
     - Scales numerical features and one-hot encodes compact categoricals.
     - Handles class imbalance using Random Undersampling (majority class reduced to 4x the minority).
  3. Saves encoding mappings for deployment.
  4. Splits the data into training and test sets.
  5. Transforms and samples the training data.
  6. Defines hyperparameter grids for Logistic Regression and XGBoost.
  7. Trains both models using grid search.
  8. Evaluates each model (accuracy, precision, recall, F1, ROC AUC) and saves ROC and confusion matrix plots.
  9. Saves the trained models and encoding mappings.

### `Train_CreditCard_model.py`

- **Purpose:**  
  Trains and evaluates models on the cleaned credit card dataset.

- **Workflow:**
  1. Loads the cleaned credit card data from `../data/processed/cleaned_creditcard_data.csv`.
  2. Initializes the `FraudPreprocessor` in `creditcard_data` mode, which:
     - Scales all numeric features (V1–V28, Amount).
     - Handles severe class imbalance using SMOTE oversampling.
  3. Splits the data into training and test sets.
  4. Transforms and applies SMOTE to the training data.
  5. Defines hyperparameter grids for Logistic Regression and XGBoost.
  6. Trains both models using grid search.
  7. Evaluates each model (accuracy, precision, recall, F1, ROC AUC) and saves ROC and confusion matrix plots.
  8. Saves the trained models.

---

## Outputs

- Trained model files (`.pkl`) saved in `../models/Fraud Model/` and `../models/CreditCard Model/`.
- Evaluation plots (ROC curves, confusion matrices) saved in the corresponding `plots/` subdirectories.
- Encoding mappings for fraud data saved for deployment.

---

## Usage

Run each script from the project root or the `scripts/` directory:

```bash
python scripts/Train_Fraud_model.py
python scripts/Train_CreditCard_model.py
```

Ensure all dependencies are installed and the processed data files are available.