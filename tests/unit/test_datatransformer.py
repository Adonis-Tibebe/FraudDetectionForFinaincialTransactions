import pandas as pd
from src.core.DataTransformer import FraudPreprocessor

def test_fraud_preprocessor_fit_transform():
    # Minimal fraud-like data
    df = pd.DataFrame({
        "device_id": ["dev1", "dev2", "dev1", "dev3"],
        "country": ["US", "UK", "US", "CA"],
        "purchase_value": [100, 200, 150, 120],
        "age": [30, 40, 22, 35],
        "purchase_hour": [10, 12, 10, 14],
        "purchase_dayofweek": [1, 2, 1, 3],
        "time_since_signup": [2.5, 3.0, 1.0, 4.0],
        "user_txn_count": [1, 2, 1, 1],
        "user_txn_velocity": [0.5, 1.0, 0.5, 2.0],
        "source": ["web", "app", "web", "app"],
        "browser": ["chrome", "safari", "chrome", "firefox"],
        "sex": ["M", "F", "M", "F"]
    })
    y = [0, 1, 0, 1]

    pre = FraudPreprocessor(mode="fraud_data")
    pre.fit(df, y)
    X_trans = pre.transform(df)
    assert isinstance(X_trans, pd.DataFrame)
    assert X_trans.shape[0] == 4
    print("✅ test_fraud_preprocessor_fit_transform passed.")

if __name__ == "__main__":
    test_fraud_preprocessor_fit_transform()