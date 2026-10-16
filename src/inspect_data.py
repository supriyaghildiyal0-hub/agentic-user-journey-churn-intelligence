
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Locate the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "events.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Check that events.csv is inside data/raw."
    )

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# 3. Inspect sample records
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# 4. Check column names and data types
print("\n========== COLUMN INFORMATION ==========")
print(df.info())

# 5. Check missing values
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 6. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Exact duplicate rows:", df.duplicated().sum())

# 7. Analyze event types
print("\n========== EVENT DISTRIBUTION ==========")
print(df["event"].value_counts(dropna=False))

print("\nEvent percentages:")
print(
    (df["event"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)

# 8. Analyze visitors and products
print("\n========== CUSTOMER ACTIVITY ==========")
print("Unique visitors:", df["visitorid"].nunique())
print("Unique products:", df["itemid"].nunique())

# 9. Analyze transactions
if "transactionid" in df.columns:
    print(
        "Rows with transaction IDs:",
        df["transactionid"].notna().sum()
    )
    print(
        "Unique transaction IDs:",
        df["transactionid"].nunique()
    )

# 10. Inspect timestamps
print("\n========== TIMESTAMP ANALYSIS ==========")
timestamps = pd.to_numeric(df["timestamp"], errors="coerce")

print("Invalid or missing timestamps:", timestamps.isna().sum())

valid_timestamps = timestamps.dropna()

if not valid_timestamps.empty:
    dates = pd.to_datetime(
        valid_timestamps,
        unit="ms",
        errors="coerce",
        utc=True
    )

    print("First event:", dates.min())
    print("Last event:", dates.max())

print("\n========== INSPECTION COMPLETE ==========")
