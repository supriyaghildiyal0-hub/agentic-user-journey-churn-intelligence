
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "events_clean.csv.gz"
OUTPUT_DIR = ROOT / "data" / "processed"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

print("Loading cleaned events...")
df = pd.read_csv(DATA_PATH)

df["event_datetime"] = pd.to_datetime(
    df["event_datetime"],
    format="mixed",
    utc=True,
    errors="raise"
)

print("\n========== CUSTOMER JOURNEY ANALYTICS ==========")

# 1. Unique visitors for each event type
event_visitors = (
    df.groupby("event")["visitorid"]
    .nunique()
    .reindex(["view", "addtocart", "transaction"], fill_value=0)
)

print("\n--- Unique visitors by event ---")
print(event_visitors)

# 2. Visitor-level behavior flags
visitor_features = df.pivot_table(
    index="visitorid",
    columns="event",
    values="timestamp",
    aggfunc="size",
    fill_value=0
)

for event in ["view", "addtocart", "transaction"]:
    if event not in visitor_features.columns:
        visitor_features[event] = 0

visitor_features = visitor_features[
    ["view", "addtocart", "transaction"]
].copy()

visitor_features.columns = [
    "view_events",
    "cart_events",
    "transaction_events"
]

visitor_features["viewed_product"] = (
    visitor_features["view_events"] > 0
)

visitor_features["added_to_cart"] = (
    visitor_features["cart_events"] > 0
)

visitor_features["made_transaction"] = (
    visitor_features["transaction_events"] > 0
)

# 3. Event-based visitor funnel
funnel = pd.DataFrame({
    "stage": [
        "Viewed product",
        "Added to cart",
        "Completed transaction"
    ],
    "unique_visitors": [
        int(visitor_features["viewed_product"].sum()),
        int(visitor_features["added_to_cart"].sum()),
        int(visitor_features["made_transaction"].sum())
    ]
})

funnel["percentage_of_all_visitors"] = (
    funnel["unique_visitors"]
    / visitor_features.shape[0] * 100
).round(2)

print("\n--- Visitor reach funnel ---")
print(funnel.to_string(index=False))

# 4. Visitor overlap between stages
print("\n--- Visitor behavior overlap ---")

overlap = {
    "view_and_cart": int(
        (
            visitor_features["viewed_product"]
            & visitor_features["added_to_cart"]
        ).sum()
    ),
    "cart_and_transaction": int(
        (
            visitor_features["added_to_cart"]
            & visitor_features["made_transaction"]
        ).sum()
    ),
    "view_and_transaction": int(
        (
            visitor_features["viewed_product"]
            & visitor_features["made_transaction"]
        ).sum()
    ),
    "all_three_events": int(
        (
            visitor_features["viewed_product"]
            & visitor_features["added_to_cart"]
            & visitor_features["made_transaction"]
        ).sum()
    )
}

for name, count in overlap.items():
    print(f"{name}: {count:,}")

# 5. Visitor activity segments
visitor_features["activity_segment"] = "Single-event visitor"

visitor_features.loc[
    visitor_features["view_events"]
    + visitor_features["cart_events"]
    + visitor_features["transaction_events"] > 1,
    "activity_segment"
] = "Multiple-event visitor"

visitor_features.loc[
    visitor_features["made_transaction"],
    "activity_segment"
] = "Visitor with transaction"

print("\n--- Visitor activity segments ---")
print(visitor_features["activity_segment"].value_counts())

# 6. Daily activity trends
df["event_date"] = df["event_datetime"].dt.strftime("%Y-%m-%d")

daily_activity = (
    df.groupby(["event_date", "event"])
    .agg(
        event_count=("visitorid", "size"),
        unique_visitors=("visitorid", "nunique")
    )
    .reset_index()
)

# 7. Save outputs
funnel.to_csv(OUTPUT_DIR / "journey_funnel.csv", index=False)

visitor_features.reset_index().to_csv(
    OUTPUT_DIR / "visitor_features.csv.gz",
    index=False,
    compression="gzip"
)

daily_activity.to_csv(
    OUTPUT_DIR / "daily_activity.csv",
    index=False
)

print("\n========== ANALYSIS SUMMARY ==========")
print("Total unique visitors:", f"{len(visitor_features):,}")
print("Funnel file saved: journey_funnel.csv")
print("Visitor features saved: visitor_features.csv.gz")
print("Daily trends saved: daily_activity.csv")
print("\nCustomer journey analysis completed!")
