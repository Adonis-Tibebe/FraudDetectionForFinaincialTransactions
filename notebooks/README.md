# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis (EDA), feature engineering, and demonstration of the data processing pipeline for financial fraud detection.

## Contents

- `EDA_And_Data_Cleanup.ipynb`:
  - Performs in-depth exploratory data analysis on both the credit card and fraud datasets.
  - Includes data loading, inspection, summary statistics, and visualizations (distribution plots, bar plots, violin plots, etc.).
  - Handles initial data cleaning, such as removing nulls, duplicates, and converting time columns to datetime.
  - Maps IP addresses to countries using the provided mapping file.

- `Feature_engineering_and_Transformation.ipynb`:
  - Demonstrates feature engineering for the fraud dataset (e.g., time-based features, transaction frequency, velocity).
  - Shows how to preprocess both datasets using the custom `FraudPreprocessor` from `src/core/DataTransformer.py`.
  - Explains and applies class imbalance handling (Random Undersampling for categorical-rich data, SMOTE for numeric data).
  - Saves feature-engineered datasets for downstream modeling.

## Usage

These notebooks are intended for interactive exploration and demonstration of the data pipeline. To run them:

1. Ensure all dependencies are installed (see `requirements.txt`).
2. Make sure the required data files are present in the `data/` directory (use `dvc pull` if needed).
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open and run the notebooks in order for a full walkthrough of the data preparation process.

## Notes
- The notebooks import utility and transformation functions from the `src/` directory for modularity and reusability.
- For production or automated workflows, refer to the code in `src/` rather than copying notebook code.
