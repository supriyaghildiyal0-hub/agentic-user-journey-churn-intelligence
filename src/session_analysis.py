
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "events_clean.csv.gz"
OUTPUT_DIR = ROOT / "data" / "processed"

print("Loading cleaned events...")
df = pd.read_csv(DATA_PATH)

# Parse timestamps consistently
df["event_datetime"] = pd.to_datetime(
    df["event_datetime"],
    format="mixed",
    utc=True,
    errors="raise"
)

# 1. Sort each visitor's events chronologically
df = df.sort_values(
    ["visitorid", "event_datetime"]
).reset_index(drop=True)

# 2. Create sessions using a 30-minute inactivity threshold
gap = df.groupby("visitorid")["event_datetime"].diff()

new_session = gap.isna() | (gap > pd.Timedelta(minutes=30))

df["session_number"] = (
    new_session.groupby(df["visitorid"]).cumsum().astype("int32")
)

# Unique session identifier
df["session_id"] = (
    df["visitorid"].astype(str)
    + "_"
    + df["session_number"].astype(str)
)

print("\n========== SESSION SUMMARY ==========")
print("Total events:", f"{len(df):,}")
print("Unique visitors:", f"{df['visitorid'].nunique():,}")
print("Unique sessions:", f"{df['session_id'].nunique():,}")

# 3. Summarize events within each session
session_features = df.pivot_table(
    index=["visitorid", "session_id"],
    columns="event",
    values="timestamp",
    aggfunc="size",
    fill_value=0
)

for event in ["view", "addtocart", "transaction"]:
    if event not in session_features.columns:
        session_features[event] = 0

session_features = session_features[
    ["view", "addtocart", "transaction"]
].copy()

session_features.columns = [
    "view_events", "cart_events", "transaction_events"
]

# 4. Determine which stages occurred in each session
session_features["viewed"] = session_features["view_events"] > 0
session_features["added_to_cart"] = session_features["cart_events"] > 0
session_features["purchased"] = (
    session_features["transaction_events"] > 0
)

# 5. Ordered stage-presence funnel
# Each stage must be present in the same session.
# Timestamp ordering will be validated separately below.
session_features["view_then_cart"] = False
session_features["cart_then_purchase"] = False
session_features["view_cart_purchase"] = False

# Extract first occurrence times for each event per session
first_times = df.pivot_table(
    index=["visitorid", "session_id"],
    columns="event",
    values="event_datetime",
    aggfunc="min"
)

for event in ["view", "addtocart", "transaction"]:
    if event not in first_times.columns:
        first_times[event] = pd.NaT

view_time = first_times["view"]
cart_time = first_times["addtocart"]
purchase_time = first_times["transaction"]

session_features["view_then_cart"] = (
    view_time.notna()
    & cart_time.notna()
    & (view_time < cart_time)
)

session_features["cart_then_purchase"] = (
    cart_time.notna()
    & purchase_time.notna()
    & (cart_time < purchase_time)
)

session_features["view_cart_purchase"] = (
    session_features["view_then_cart"]
    & session_features["cart_then_purchase"]
)

# 6. Report session funnel
total_sessions = len(session_features)

funnel = pd.DataFrame({
    "stage": [
        "Sessions with product views",
        "Sessions with cart additions",
        "Sessions with transactions",
        "Sessions with view before cart",
        "Sessions with cart before purchase",
        "Sessions completing ordered journey"
    ],
    "sessions": [
        int(session_features["viewed"].sum()),
        int(session_features["added_to_cart"].sum()),
        int(session_features["purchased"].sum()),
        int(session_features["view_then_cart"].sum()),
        int(session_features["cart_then_purchase"].sum()),
        int(session_features["view_cart_purchase"].sum())
    ]
})

funnel["percentage_of_all_sessions"] = (
    funnel["sessions"] / total_sessions * 100
).round(2)

print("\n========== SESSION FUNNEL ==========")
print(funnel.to_string(index=False))

# 7. Save outputs
session_features.reset_index().to_csv(
    OUTPUT_DIR / "session_features.csv.gz",
    index=False,
    compression="gzip"
)

funnel.to_csv(
    OUTPUT_DIR / "session_funnel.csv",
    index=False
)

print("\nSaved session_features.csv.gz")
print("Saved session_funnel.csv")
print("\nSession analysis completed!")
