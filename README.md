# Fraud Detection for Financial Transactions

A robust, modular pipeline for detecting fraudulent financial transactions using real-world datasets. This project demonstrates best practices in data cleaning, feature engineering, preprocessing, and class imbalance handling, with reusable components for experimentation and deployment.

## Project Structure

```
FraudDetectionForFinaincialTransactions/
│
├── config/         # Configuration settings
├── data/           # Raw and processed data (DVC-tracked)
├── docs/           # Project documentation
├── examples/       # Example usage scripts (template)
├── notebooks/      # Jupyter notebooks for EDA and pipeline demos
├── scripts/        # Training scripts for model automation
├── src/            # Source code (core logic, models, utils)
├── tests/          # Unit and integration tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Features

- **Data Versioning:** Uses DVC to track raw and processed datasets for reproducibility.
- **EDA & Visualization:** Notebooks for in-depth exploratory analysis and visualization.
- **Feature Engineering:** Temporal, behavioral, and categorical feature creation (e.g., transaction frequency, velocity, time-based features).
- **Preprocessing Pipelines:** Modular transformers for both categorical-rich and fully-numeric datasets, implemented in `src/core/DataTransformer.py`.
- **Imbalance Handling:** Random Undersampling (RUS) for categorical data, SMOTE for numeric data.
- **Reusable Components:** Core logic in `src/` for easy integration and deployment.
- **Utility Functions:** Data loading, cleaning, and visualization helpers in `src/utils/utils.py`.

## Installation

1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd FraudDetectionForFinaincialTransactions
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Set up DVC and pull data:
    ```bash
    pip install dvc
    dvc pull
    ```

## Usage

- **Jupyter Notebooks:**  
  Run the notebooks in the `notebooks/` directory for EDA and pipeline demonstrations. These provide a step-by-step walkthrough of the data preparation and feature engineering process.

- **Training Scripts:**  
  The `scripts/` directory contains automated scripts for model training and evaluation:
  - `Train_Fraud_model.py`: Trains and evaluates models on the feature-engineered fraud dataset, including preprocessing, class balancing (Random Undersampling), hyperparameter tuning, evaluation, and artifact saving.
  - `Train_CreditCard_model.py`: Trains and evaluates models on the cleaned credit card dataset, including preprocessing, class balancing (SMOTE), hyperparameter tuning, evaluation, and artifact saving.
  - Both scripts support Logistic Regression and XGBoost, and save trained models and evaluation plots for deployment or further analysis.

  To run the scripts:
  ```bash
  python scripts/Train_Fraud_model.py
  python scripts/Train_CreditCard_model.py
  ```

- **Source Code:**  
  Import and use the core data transformation logic from `src/core/DataTransformer.py` and utility functions from `src/utils/utils.py` in your own scripts or pipelines. Example usage is shown in the notebooks.

- **Model Training and Selection:**  
  The pipeline supports both Logistic Regression and XGBoost models, with hyperparameter tuning and evaluation. The final model is selected based on F1, ROC AUC, and recall, with XGBoost chosen for deployment due to its superior performance on these metrics.

## Data

- Place raw data in `data/raw/` (DVC-tracked).
- Processed and feature-engineered data will be saved in `data/processed/`.
- Data files are tracked using DVC. To obtain the actual data files, run:
  ```bash
  dvc pull
  ```
- See `data/README.md` for more details.

## Notebooks

- See `notebooks/README.md` for an overview of available notebooks and their purposes.
- - Notebooks demonstrate the full workflow: EDA, cleaning, feature engineering, transformation, class balancing, and model selection.

## Configuration

- Project configuration can be managed in `config/settings.py`.
- Adjust settings as needed for your environment or experiments.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements or bug fixes.

## License

None
