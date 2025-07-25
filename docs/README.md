# Documentation

## Project Overview

**Fraud Detection for Financial Transactions** is a robust, modular pipeline developed by Adey Innovations Inc. to detect fraudulent activity in both e-commerce and banking transactions. The system is designed to balance security and user experience, leveraging advanced data science, explainable AI, and best engineering practices to deliver real-time, trustworthy fraud detection.

---

## Business Context

Fraudulent transactions threaten both revenue and customer trust. Our models are built to minimize both false positives (blocking real users) and false negatives (letting fraud through), supporting real-time monitoring, operational transparency, and regulatory compliance. The pipeline is designed for seamless integration into production systems, with a focus on explainability for business stakeholders and auditors.

---

## Data Sources

- **E-Commerce Fraud Data:**  
  Contains user, device, browser, geolocation, and behavioral features, with timestamps for signup and purchase.
- **Credit Card Transactions:**  
  Anonymized, PCA-transformed features (V1–V28), transaction amount, and time. Highly imbalanced with very few fraud cases.

All data is tracked and versioned using DVC for reproducibility.

---

## Workflow Summary

### 1. Data Loading & Cleaning

- Data is loaded from DVC-tracked sources.
- Cleaning includes null/duplicate removal, type conversion, and IP-to-country mapping.
- Outputs are saved in `data/processed/`.

### 2. Exploratory Data Analysis (EDA)

- Visualizations (distributions, bar/violin plots) reveal fraud patterns by time, geography, device, and behavior.
- Class imbalance is quantified to inform downstream modeling.
- Insights guide feature engineering and business understanding.

### 3. Feature Engineering

- **Temporal:** Purchase hour, day of week, time since signup.
- **Behavioral:** User transaction count and velocity.
- **Categorical:** Frequency encoding (device_id), target encoding (country), one-hot encoding (browser, source).
- Features are engineered to capture fraudster tactics and user behavior.

### 4. Preprocessing Pipelines

- Encapsulated in `FraudPreprocessor` (see `src/core/DataTransformer.py`).
- Handles encoding, scaling, and class imbalance (RUS for e-commerce, SMOTE for credit card).
- Saves mappings for consistent deployment.

### 5. Model Training & Evaluation

- **Models:** Logistic Regression (baseline), XGBoost (final selection).
- **Hyperparameter Tuning:** Grid search on F1 score.
- **Metrics:** Precision, recall, F1, ROC AUC (accuracy is not used due to class imbalance).
- **Scripts:** Automated in `scripts/Train_Fraud_model.py` and `scripts/Train_CreditCard_model.py`.
- **Outputs:** Trained models, evaluation plots, and mappings are saved for deployment.

### 6. Model Selection

- XGBoost selected for both datasets due to superior F1 and ROC AUC, balancing fraud detection with minimal false positives.
- Model selection rationale and metrics are documented in the notebooks and examples.

### 7. Explainability

- SHAP (SHapley Additive exPlanations) used for both global and local model interpretation.
- **Global:** Feature importance plots show which features drive fraud predictions.
- **Local:** Force plots explain individual predictions, supporting transparency and compliance.
- Insights are documented in `fraud-model-interpretatoin.ipynb` and `creditcard-model-interpretatoin .ipynb`.

### 8. Testing & CI/CD

- Unit tests for data transformation and utilities in `tests/unit/`.
- CI/CD pipeline (GitHub Actions) installs dependencies and runs tests on Windows.
- Code formatting and linting enforced via `Makefile` and `pyproject.toml`.

---

## Project Structure


---

## Key Artifacts

- **Feature-Engineered Datasets:**  
  Saved in `data/processed/Feature_engineered/` for fraud data and `data/processed/` for credit card data.
- **Trained Models:**  
  Saved in `models/Fraud Model/` and `models/CreditCard Model/`.
- **Evaluation Plots:**  
  ROC curves and confusion matrices in `models/*/plots/`.
- **Encoding Mappings:**  
  For deployment, saved in `models/Fraud Model/Mappings/`.
- **Notebooks:**  
  Full workflow, EDA, and SHAP explainability in `notebooks/`.

---

## How to Run

1. **Install dependencies:**  
   `make install` or `pip install -r requirements.txt`
2. **Pull data:**  
   `dvc pull`
3. **Run notebooks:**  
   `jupyter notebook notebooks/`
4. **Train models:**  
   ```bash
   python scripts/Train_Fraud_model.py
   python scripts/Train_CreditCard_model.py
5. **Run tests:**
    make test or pytest tests/

## Business Impact
   - Reduced Fraud Losses:
    High recall ensures most fraud is caught.
   - Improved User Experience:
    High precision minimizes false positives, reducing customer friction.
   - Operational Transparency:
    SHAP explainability supports analyst workflows and regulatory audits.
   - Scalability:
    Modular, tested codebase ready for production deployment and future enhancements.

## Limitations & Future Work
- Data drift monitoring and retraining needed for evolving fraud tactics.
- Further interpretability for PCA features in credit card data.
- Latency optimization for real-time scoring.
- Integration of analyst feedback and external threat intelligence.

## References
- DVC Documentation
- SHAP Documentation
- XGBoost Documentation

``````For further details, see the main README.md, notebooks, and scripts. ``````