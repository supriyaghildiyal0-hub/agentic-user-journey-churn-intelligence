
from pathlib import Path
import pandas as pd
import json

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "events.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "events_clean.csv.gz"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)

original_rows = len(df)

# 1. Validate required columns
required = [
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid"
]

missing_columns = set(required) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 2. Remove exact duplicate rows from the working copy
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# 3. Validate event types
allowed_events = {"view", "addtocart", "transaction"}

unexpected_events = sorted(
    set(df["event"].dropna().unique()) - allowed_events
)

if df["event"].isna().any():
    raise ValueError("Missing event types detected.")

if unexpected_events:
    raise ValueError(f"Unexpected event types: {unexpected_events}")

# 4. Validate identifiers
if df[["timestamp", "visitorid", "itemid"]].isna().any().any():
    raise ValueError("Required timestamp or identifier is missing.")

if (df["timestamp"] < 0).any():
    raise ValueError("Negative timestamps detected.")

# 5. Convert timestamps from Unix milliseconds to UTC
df["event_datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True,
    errors="coerce"
)

if df["event_datetime"].isna().any():
    raise ValueError("Timestamp conversion failed.")

# 6. Add useful time-based features
df["event_date"] = df["event_datetime"].dt.date.astype(str)
df["event_hour"] = df["event_datetime"].dt.hour
df["event_day_of_week"] = df["event_datetime"].dt.dayofweek

# 7. Create a data quality report
report = {
    "original_rows": int(original_rows),
    "duplicate_rows_removed": duplicate_count,
    "cleaned_rows": int(len(df)),
    "unique_visitors": int(df["visitorid"].nunique()),
    "unique_products": int(df["itemid"].nunique()),
    "unique_transactions": int(df["transactionid"].nunique()),
    "missing_values": {
        col: int(count)
        for col, count in df.isna().sum().items()
    },
    "event_counts": {
        str(event): int(count)
        for event, count in df["event"].value_counts().items()
    },
    "first_event_utc": str(df["event_datetime"].min()),
    "last_event_utc": str(df["event_datetime"].max())
}

# 8. Save cleaned data and report
df.to_csv(OUTPUT_PATH, index=False, compression="gzip")

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n========== CLEANING SUMMARY ==========")
print("Original rows:", original_rows)
print("Duplicate rows removed:", duplicate_count)
print("Cleaned rows:", len(df))
print("Output:", OUTPUT_PATH)
print("Quality report:", REPORT_PATH)
print("\nEvent distribution:")
print(df["event"].value_counts())
print("\nMissing values:")
print(df.isna().sum())
print("\nCleaning completed successfully.")
