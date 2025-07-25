import pandas as pd
import logging
import json
import os
import sys

sys.path.append(os.path.abspath("../"))
from src.models.model_trainer import ModelTrainer
from src.utils.utils import load_data
from src.utils.training_and_evaluation_utils import (
    train_test_split_data,
    evaluate_model,
    plot_confusion_matrix
)
from src.core.DataTransformer import FraudPreprocessor

# -------------------------
# ✅ Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# ✅ Ensure output folders exist
# -------------------------
os.makedirs("../models/Fraud Model", exist_ok=True)
os.makedirs("../models/Fraud Model/plots", exist_ok=True)
os.makedirs("../models/Fraud Model/Mappings", exist_ok=True)

models_dir = "../models/Fraud Model"
plot_dir = "../models/Fraud Model/plots"
mappings_dir = "../models/Fraud Model/Mappings"

# -------------------------
# ✅ Load fraud dataset
# -------------------------
logger.info("Loading fraud_data.csv")
df = load_data("../data/processed/Feature_engineered/Feature_engineered_fraud_data.csv")
X = df.drop(columns="class")
y = df["class"]

# -------------------------
# ✅ Initialize and fit preprocessor
# -------------------------
logger.info("Initializing FraudPreprocessor")
preprocessor = FraudPreprocessor(mode="fraud_data", sampler="auto")
preprocessor.fit(X, y)

# -------------------------
# ✅ Save encoding mappings
# -------------------------
logger.info("Saving encoding maps for deployment")
mappings = preprocessor.save_mappings()
with open(f"{mappings_dir}/fraud_encoding_maps.json", "w") as f:
    json.dump(mappings, f)

# -------------------------
# ✅ Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split_data(X, y, stratify=y)

# -------------------------
# ✅ Transform + sampling (RUS)
# -------------------------
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
X_train_sampled, y_train_sampled = preprocessor.sample(X_train_transformed, y_train)

# -------------------------
# ✅ Parameter grids
# -------------------------
param_grids = {
    "logistic_regression": {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["liblinear"]
    },
    "xgboost": {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }
}

# -------------------------
# ✅ Train models
# -------------------------
models = {
    "logistic_regression": ModelTrainer("logistic_regression"),
    "xgboost": ModelTrainer("gbm")
}

for name, trainer in models.items():
    logger.info(f"⚙️ Training model: {name}")
    param_grid = param_grids.get(name)
    trainer.train(X_train_sampled, y_train_sampled, param_grid=param_grid, search_type="grid")

    y_pred = trainer.predict(X_test_transformed)
    y_proba = trainer.predict_proba(X_test_transformed)

    # Evaluate
    evaluate_model(
        y_test, y_pred, y_proba,
        model_name=name,
        save_roc_path=f"{plot_dir}/roc_{name}.png"
    )
    plot_confusion_matrix(
        y_test, y_pred,
        model_name=name,
        save_path=f"{plot_dir}/cm_{name}.png"
    )

    # Save model
    trainer.save_model(f"{models_dir}/{name}_fraud.pkl")

logger.info("✅ All fraud models trained.")
logger.info("✅ All fraud models evaluated.")
logger.info("✅ All fraud models saved.")
logger.info("✅ Encoding mappings for dataset saved.")