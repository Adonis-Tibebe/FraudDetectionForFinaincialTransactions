# Examples

This directory contains example workflows for the fraud detection project.

---

## Notebook: `Notebook_simulation_of_Training_scripts.ipynb`

This notebook demonstrates a full end-to-end simulation of the model training and evaluation pipeline for both the **fraud** and **credit card** datasets. The workflow is designed to mirror the scripts in the `scripts/` directory, but is run in a notebook environment (Kaggle) due to computational constraints.

### Workflow Overview

1. **Environment Setup**
   - Installs all required dependencies (e.g., `xgboost`, `scikit-learn`, `imbalanced-learn`).
   - Sets up logging for reproducibility and debugging.

2. **Pipeline Components**
   - Defines the `FraudPreprocessor` class for feature engineering, encoding, scaling, and class balancing.
   - Defines the `ModelTrainer` class for model initialization, training (with hyperparameter tuning), prediction, and saving.
   - Defines utility functions for data splitting and evaluation (including ROC and confusion matrix plotting).

3. **Fraud Dataset Pipeline**
   - Loads the feature-engineered fraud dataset.
   - Applies the `FraudPreprocessor` in `fraud_data` mode:
     - Frequency encoding for `device_id`
     - Target encoding for `country`
     - Standard scaling for numerical features
     - One-hot encoding for compact categoricals
     - Random Undersampling (RUS) to address class imbalance (majority class reduced to 4x the minority)
   - Splits data into train/test, transforms, and samples.
   - Trains both Logistic Regression and XGBoost models with grid search.
   - Evaluates models using accuracy, precision, recall, F1, and ROC AUC.
   - Plots and saves evaluation metrics and trained models.

4. **Credit Card Dataset Pipeline**
   - Loads the cleaned credit card dataset.
   - Applies the `FraudPreprocessor` in `creditcard_data` mode:
     - Standard scaling for all numeric features (V1–V28, Amount)
     - SMOTE oversampling to address severe class imbalance
   - Splits data into train/test, transforms, and applies SMOTE.
   - Trains both Logistic Regression and XGBoost models with grid search.
   - Evaluates models using the same metrics as above.
   - Plots and saves evaluation metrics and trained models.

---

## Model Selection Justification

After evaluating both **Logistic Regression** and **XGBoost** on both datasets, the notebook provides a clear, metric-based justification for selecting XGBoost as the final model for deployment.

### Fraud Dataset (Random Undersampling, RUS = 0.25)

| Metric         | Logistic Regression | XGBoost     |
|----------------|---------------------|-------------|
| Accuracy       | 0.9510              | **0.9554**  |
| Precision      | 0.9125              | **0.9980**  |
| Recall         | 0.5272              | 0.5244      |
| F1 Score       | 0.6683              | **0.6875**  |
| ROC AUC        | 0.8368              | **0.8421**  |

**Justification:**  
While recall is similar, XGBoost achieves much higher **precision**, **F1 Score**, and **ROC AUC**, indicating better discrimination and fewer false positives. This is crucial in fraud detection, where minimizing false alarms while catching as many frauds as possible is key.

---

### Credit Card Dataset (SMOTE Oversampling)

| Metric         | Logistic Regression | XGBoost     |
|----------------|---------------------|-------------|
| Accuracy       | 0.9511              | **0.9512**  |
| Precision      | 0.9115              | 0.8513      |
| Recall         | 0.5279              | **0.5784**  |
| F1 Score       | 0.6686              | **0.6888**  |
| ROC AUC        | 0.8376              | **0.8430**  |

**Justification:**  
XGBoost outperforms Logistic Regression in **Recall**, **F1 Score**, and **ROC AUC**. While Logistic Regression has slightly higher precision, XGBoost's higher recall and F1 are more important for fraud detection, where missing fraudulent transactions is costlier than flagging non-fraud ones.

---

### 📌 Conclusion

Across both datasets, **XGBoost** consistently delivers better or comparable results, especially in **Recall**, **F1 Score**, and **ROC AUC**—the most critical metrics for imbalanced fraud detection tasks. Therefore, **XGBoost** is chosen as the final model for deployment.

---

## Usage

- Use this notebook as a reference for structuring your own training and evaluation scripts.
- The notebook is designed for demonstration and may use Kaggle-specific paths or commands; adjust as needed for your environment.
- For production or automation, adapt the logic into scripts in the `scripts/` directory and use the modular code from