
from pathlib import Path
import pandas as pd

# Locate and load the original dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "events.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATA QUALITY VALIDATION ==========")

# 1. Validate required columns
required_columns = {
    "timestamp", "visitorid", "event", "itemid", "transactionid"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("\nAll required columns are present.")

# 2. Check missing values in important fields
print("\n--- Missing values ---")
print(df.isna().sum())

# 3. Investigate duplicate rows
duplicates = df[df.duplicated(keep=False)]

print("\n--- Duplicate investigation ---")
print("Rows involved in duplicate groups:", len(duplicates))
print("Extra duplicate rows:", df.duplicated().sum())

if not duplicates.empty:
    print("\nDuplicate rows by event type:")
    print(duplicates["event"].value_counts(dropna=False))

    print("\nExamples of duplicate rows:")
    print(duplicates.head(10).to_string(index=False))

# 4. Analyze transaction IDs by event type
print("\n--- Transaction ID validation ---")

transaction_summary = df.groupby("event").agg(
    total_events=("event", "size"),
    rows_with_transaction_id=("transactionid", "count"),
    unique_visitors=("visitorid", "nunique"),
    unique_products=("itemid", "nunique")
)

print(transaction_summary)

transaction_rows = df[df["transactionid"].notna()]

print(
    "\nUnique transaction IDs:",
    transaction_rows["transactionid"].nunique()
)

print(
    "Unique visitors with transaction IDs:",
    transaction_rows["visitorid"].nunique()
)

# 5. Validate timestamp range
print("\n--- Timestamp validation ---")

ts = pd.to_numeric(df["timestamp"], errors="coerce")

print("Missing or invalid timestamps:", ts.isna().sum())
print("Minimum raw timestamp:", ts.min())
print("Maximum raw timestamp:", ts.max())

valid_ts = pd.to_datetime(
    ts,
    unit="ms",
    errors="coerce",
    utc=True
)

print("Unconvertible timestamps:", valid_ts.isna().sum())
print("Earliest event:", valid_ts.min())
print("Latest event:", valid_ts.max())

# 6. Check missing visitor and product identifiers
print("\n--- Identifier validation ---")
print("Missing visitor IDs:", df["visitorid"].isna().sum())
print("Missing product IDs:", df["itemid"].isna().sum())

# 7. Understand how active visitors behave
print("\n--- Visitor activity ---")

events_per_visitor = df.groupby("visitorid").size()

print("Average events per visitor:", round(events_per_visitor.mean(), 2))
print("Median events per visitor:", events_per_visitor.median())
print("Visitors with only one event:", (events_per_visitor == 1).sum())
print("Visitors with more than 10 events:", (events_per_visitor > 10).sum())

print("\n========== VALIDATION COMPLETE ==========")
