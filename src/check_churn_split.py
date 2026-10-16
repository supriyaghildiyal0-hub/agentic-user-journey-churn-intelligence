
import pandas as pd

file_path = "data/processed/churn_training_data.csv.gz"

print("Loading churn dataset...")
df = pd.read_csv(file_path, compression="gzip")

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"], errors="coerce", utc=True
)

print("\nDataset shape:", df.shape)

print("\n1. Churn label distribution:")
print(df["churn_label"].value_counts(dropna=False))
print("\nChurn label proportions:")
print(df["churn_label"].value_counts(normalize=True, dropna=False).round(4))

print("\n2. Snapshot date range:")
print("Earliest:", df["snapshot_date"].min())
print("Latest:", df["snapshot_date"].max())

print("\n3. Number of visitors per snapshot date:")
print(df.groupby("snapshot_date").size().to_string())

print("\n4. Churn labels by snapshot date:")
print(
    pd.crosstab(df["snapshot_date"], df["churn_label"])
    .to_string()
)

print("\n5. Missing labels:", df["churn_label"].isna().sum())
