import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, mode='fraud_data', sampler='auto'):
        self.mode = mode
        self.sampler = sampler
        self.device_freq_map = None
        self.country_fraud_map = None
        self.global_fraud_rate = None
        self.num_cols = []
        self.cat_cols = []
        self.column_transformer = None

    def fit(self, X, y):
        logger.info(f"Fitting FraudPreprocessor for mode: {self.mode}")
        df = X.copy()
        df['class'] = y

        if self.mode == 'fraud_data':
            # Frequency encoding
            self.device_freq_map = df['device_id'].value_counts(normalize=True).to_dict()
            df['device_id_freq'] = df['device_id'].map(self.device_freq_map).fillna(0)

            # Target encoding
            self.country_fraud_map = df.groupby('country')['class'].mean().to_dict()
            self.global_fraud_rate = df['class'].mean()
            df['country_encoded'] = df['country'].map(self.country_fraud_map).fillna(self.global_fraud_rate)

            self.num_cols = ["purchase_value", 'age', 'purchase_hour', 'purchase_dayofweek',
                             'time_since_signup', 'user_txn_count', 'user_txn_velocity',
                             'device_id_freq', 'country_encoded']
            self.cat_cols = ['source', 'browser', 'sex']

        elif self.mode == 'creditcard_data':
            self.num_cols = [col for col in df.columns if col.startswith('V')] + ['Amount']
            self.cat_cols = []

        self.column_transformer = ColumnTransformer([
            ('num', StandardScaler(), self.num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), self.cat_cols)
        ])

        self.column_transformer.fit(df.drop(columns='class'))
        logger.info("ColumnTransformer fitted")

        return self

    def transform(self, X):
        logger.info("Transforming data with trained encoders/scalers")
        df = X.copy()

        if self.mode == 'fraud_data':
            df['device_id_freq'] = df['device_id'].map(self.device_freq_map).fillna(0)
            df['country_encoded'] = df['country'].map(self.country_fraud_map).fillna(self.global_fraud_rate)

        X_transformed = self.column_transformer.transform(df)
        feature_names = self.column_transformer.get_feature_names_out()
        logger.info(f"Transformation complete → shape: {X_transformed.shape}")

        return pd.DataFrame(X_transformed, columns=feature_names)

    def sample(self, X, y):
        logger.info(f"Applying sampling method: {self.sampler}")
        if self.sampler == 'auto':
            if self.mode == 'creditcard_data':
                sampler = SMOTE(random_state=42) 
            else:
                sampler = RandomUnderSampler(
                sampling_strategy=0.25,  # keep minority, reduce majority so it’s 4x larger
                random_state=42
                )
        else:
            sampler = self.sampler # If custom sampler passed

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Resampled data → new shape: {X_resampled.shape}")
        return X_resampled, y_resampled

    def save_mappings(self):
        logger.info("Saving encoding maps for deployment or later inference")
        return {
            'device_freq_map': self.device_freq_map,
            'country_fraud_map': self.country_fraud_map,
            'global_fraud_rate': self.global_fraud_rate
        }

    def transform_for_inference(self, user_dict):
        """
        Accepts a single user query (dict-like), returns transformed row for prediction
        """
        logger.info("Transforming user input for prediction")
        df = pd.DataFrame([user_dict])

        if self.mode == 'fraud_data':
            df['device_id_freq'] = df['device_id'].map(self.device_freq_map).fillna(0)
            df['country_encoded'] = df['country'].map(self.country_fraud_map).fillna(self.global_fraud_rate)

        transformed = self.column_transformer.transform(df)
        logger.info("Transformation for inference complete")
        return transformed