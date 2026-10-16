
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "events_clean.csv.gz"
OUTPUT_DIR = ROOT / "data" / "processed"

print("Loading cleaned events...")

df = pd.read_csv(DATA_PATH)

df["event_datetime"] = pd.to_datetime(
    df["event_datetime"],
    format="mixed",
    utc=True,
    errors="raise"
)

df["event_date"] = df["event_datetime"].dt.floor("D")

df = df.sort_values(
    ["visitorid", "event_datetime"]
).reset_index(drop=True)

# Experiment configuration
OBSERVATION_DAYS = 30
PREDICTION_DAYS = 14
SNAPSHOT_STEP_DAYS = 14

first_date = df["event_date"].min().normalize()
last_date = df["event_date"].max().normalize()

# The prediction window must finish before the dataset ends.
first_cutoff = first_date + pd.Timedelta(days=OBSERVATION_DAYS)
last_cutoff = (
    last_date
    - pd.Timedelta(days=PREDICTION_DAYS)
    + pd.Timedelta(days=1)
)

cutoffs = pd.date_range(
    start=first_cutoff,
    end=last_cutoff,
    freq=f"{SNAPSHOT_STEP_DAYS}D",
    tz="UTC"
)

if len(cutoffs) == 0:
    raise ValueError("Not enough data for the selected time windows.")

print("\nDataset period:", first_date, "to", last_date)
print("Number of snapshots:", len(cutoffs))

all_snapshots = []

for snapshot_number, cutoff in enumerate(cutoffs, start=1):
    observation_start = cutoff - pd.Timedelta(days=OBSERVATION_DAYS)
    prediction_end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)

    print(
        f"\nSnapshot {snapshot_number}/{len(cutoffs)}:",
        cutoff.date()
    )

    # Historical data only: features must not use future events.
    observation = df[
        (df["event_datetime"] >= observation_start)
        & (df["event_datetime"] < cutoff)
    ].copy()

    future = df[
        (df["event_datetime"] >= cutoff)
        & (df["event_datetime"] < prediction_end)
    ]

    if observation.empty:
        continue

    # Only visitors active during the observation window are eligible.
    features = observation.groupby("visitorid").agg(
        total_events=("event", "size"),
        view_events=("event", lambda s: (s == "view").sum()),
        cart_events=("event", lambda s: (s == "addtocart").sum()),
        transaction_events=("event", lambda s: (s == "transaction").sum()),
        unique_products=("itemid", "nunique"),
        active_days=("event_date", "nunique"),
        last_seen=("event_datetime", "max"),
        first_seen=("event_datetime", "min")
    )

    # Visitor-level behavioral features
    features["viewed"] = (features["view_events"] > 0).astype(int)
    features["added_to_cart"] = (features["cart_events"] > 0).astype(int)
    features["purchased"] = (
        features["transaction_events"] > 0
    ).astype(int)

    features["cart_to_view_ratio"] = (
        features["cart_events"]
        / features["view_events"].replace(0, np.nan)
    ).fillna(0)

    features["purchase_to_view_ratio"] = (
        features["transaction_events"]
        / features["view_events"].replace(0, np.nan)
    ).fillna(0)

    features["days_since_last_activity"] = (
        (cutoff - features["last_seen"]).dt.total_seconds()
        / 86400
    )

    features["days_since_first_activity"] = (
        (cutoff - features["first_seen"]).dt.total_seconds()
        / 86400
    )

    # Count sessions within the observation window.
    # A new session starts after more than 30 minutes of inactivity.
    observation = observation.sort_values(
        ["visitorid", "event_datetime"]
    )

    gap = observation.groupby("visitorid")[
        "event_datetime"
    ].diff()

    new_session = gap.isna() | (
        gap > pd.Timedelta(minutes=30)
    )

    observation["session_start"] = new_session.astype(int)

    session_counts = observation.groupby(
        "visitorid"
    )["session_start"].sum()

    features["session_count"] = session_counts

    # Label visitors using only the following 14-day period.
    future_visitors = set(future["visitorid"].unique())

    features["churn_label"] = (
        ~features.index.isin(future_visitors)
    ).astype(int)

    # Store the snapshot cutoff for temporal evaluation.
    features["snapshot_date"] = cutoff.strftime("%Y-%m-%d")

    # Drop raw timestamps; retain only engineered model features.
    features = features.drop(
        columns=["last_seen", "first_seen"]
    )

    features = features.reset_index()

    all_snapshots.append(features)

if not all_snapshots:
    raise ValueError("No eligible snapshots were generated.")

churn_df = pd.concat(
    all_snapshots,
    ignore_index=True
)

# Validate the output
print("\n========== CHURN DATASET SUMMARY ==========")
print("Rows:", f"{len(churn_df):,}")
print("Unique visitors:", f"{churn_df['visitorid'].nunique():,}")
print("Snapshots:", churn_df["snapshot_date"].nunique())

print("\nLabel distribution:")
print(churn_df["churn_label"].value_counts())

print("\nLabel percentages:")
print(
    (churn_df["churn_label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(churn_df.isna().sum())

output_path = OUTPUT_DIR / "churn_training_data.csv.gz"
churn_df.to_csv(
    output_path,
    index=False,
    compression="gzip"
)

print("\nSaved:", output_path)
print("Churn dataset generation completed!")
