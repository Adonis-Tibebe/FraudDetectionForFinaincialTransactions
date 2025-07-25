import os
import pandas as pd
from src.utils.utils import load_data, clean_data

def test_load_and_clean_data():
    # Create a small dummy CSV for testing
    test_csv = "test_data.csv"
    df = pd.DataFrame({
        "A": [1, 2, 2, None],
        "B": ["x", "y", "y", "z"],
        "time": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
    })
    df.to_csv(test_csv, index=False)

    # Test load_data
    loaded = load_data(test_csv)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape[0] == 4

    # Test clean_data (removes nulls and duplicates, converts time)
    cleaned = clean_data(loaded, time_column=["time"])
    assert cleaned.shape[0] == 2  # Only unique, non-null rows remain
    assert pd.api.types.is_datetime64_any_dtype(cleaned["time"])

    os.remove(test_csv)
    print("✅ test_load_and_clean_data passed.")

if __name__ == "__main__":
    test_load_and_clean_data()