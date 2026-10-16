
import pandas as pd

path = "data/processed/churn_training_data.csv.gz"

df = pd.read_csv(path, compression="gzip")
df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"], utc=True, errors="coerce"
)

print("Original rows:", len(df))
print("\nAll-row label counts:")
print(df["churn_label"].value_counts())

# Earliest snapshot per visitor
earliest = (
    df.sort_values("snapshot_date")
      .drop_duplicates("visitorid", keep="first")
)

print("\nEarliest snapshot per visitor:")
print(earliest["churn_label"].value_counts())

# Latest snapshot per visitor
latest = (
    df.sort_values("snapshot_date")
      .drop_duplicates("visitorid", keep="last")
)

print("\nLatest snapshot per visitor:")
print(latest["churn_label"].value_counts())

print("\nLatest snapshot dates:")
print(latest["snapshot_date"].value_counts().sort_index())
