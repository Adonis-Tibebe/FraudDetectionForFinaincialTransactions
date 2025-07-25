import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        self.model = self._init_model()
        logger.info(f"Initialized model: {self.model_name}")

    def _init_model(self):
        if self.model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        elif self.model_name == "random_forest":
            return RandomForestClassifier()
        elif self.model_name == "gbm":
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def train(self, X_train, y_train, param_grid=None, search_type="grid"):
        logger.info(f"Training model: {self.model_name}")
        if param_grid:
            logger.info(f"Running hyperparameter search: {search_type}")
            if search_type == "grid":
                search = GridSearchCV(self.model, param_grid, cv=2, scoring="roc_auc")
            else:
                search = RandomizedSearchCV(self.model, param_grid, cv=2, scoring="roc_auc", n_iter=10)
            search.fit(X_train, y_train)
            logger.info(f"Best parameters for {self.model_name}: {search.best_params_}")
            self.model = search.best_estimator_
            logger.info(f"Best model selected: {self.model}")
            return search
        else:
            self.model.fit(X_train, y_train)
            logger.info("Training complete without search")
            return self.model

    def predict(self, X):
        logger.info("Running predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        logger.info("Calculating prediction probabilities")
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, filepath):
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self.model, filepath)
        return filepath