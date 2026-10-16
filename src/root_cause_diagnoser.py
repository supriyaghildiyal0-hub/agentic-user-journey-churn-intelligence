
from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "churn_training_data.csv.gz"
)

MODEL_PATH = ROOT / "models" / "random_forest.joblib"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "visitor_root_cause_report.csv"

print("Loading visitor data and trained model...")

# --------------------------------------------------
# 2. Load data and model
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="raise"
)

model = joblib.load(MODEL_PATH)

# Use the latest available snapshot for each visitor.
# This represents the most recent feature record available
# in this historical dataset, not a live prediction.

latest_df = (
    df.sort_values("snapshot_date")
    .drop_duplicates(subset="visitorid", keep="last")
    .copy()
    .reset_index(drop=True)
)
# Test on a smaller sample first
latest_df = latest_df.sample(
    n=min(50000, len(latest_df)),
    random_state=42
).copy()
print("Latest snapshot:", latest_df["snapshot_date"].max().date())
print("Visitors to analyze:", f"{len(latest_df):,}")

# --------------------------------------------------
# 3. Prepare model features
# --------------------------------------------------

target = "churn_label"

excluded = {
    "visitorid",
    "snapshot_date",
    target
}

feature_cols = [
    col for col in latest_df.columns
    if col not in excluded
]

X = latest_df[feature_cols].copy()

print("Generating inactivity-risk predictions...")

# Probability of class 1:
# no recorded event during the following 14 days.
probabilities = model.predict_proba(X)

classes = list(model.named_steps["model"].classes_)
class_1_index = classes.index(1)

latest_df["inactivity_risk_score"] = (
    probabilities[:, class_1_index]
)

latest_df["predicted_inactive"] = (
    model.predict(X) == 1
).astype(int)

# --------------------------------------------------
# 4. Root-cause diagnosis rules
# --------------------------------------------------

def diagnose_visitor(row):
    """
    Generate a transparent, rule-based explanation.
    These are behavioral indicators, not proven causes.
    """

    reasons = []
    actions = []

    days_inactive = row["days_since_last_activity"]
    active_days = row["active_days"]
    sessions = row["session_count"]
    views = row["view_events"]
    carts = row["cart_events"]
    transactions = row["transaction_events"]

    # Rule 1: Recent activity gap
    if days_inactive >= 7:
        reasons.append(
            "Long gap since last recorded activity"
        )
        actions.append(
            "Consider a permission-based re-engagement message"
        )

    # Rule 2: Low engagement
    if active_days <= 1 and sessions <= 1:
        reasons.append(
            "Low engagement: activity occurred in very few sessions"
        )
        actions.append(
            "Recommend relevant products or useful discovery content"
        )

    # Rule 3: Low browsing activity
    if views <= 2:
        reasons.append(
            "Very limited product browsing"
        )
        actions.append(
            "Improve product discovery and personalize recommendations"
        )

    # Rule 4: Cart abandonment indicator
    if carts > 0 and transactions == 0:
        reasons.append(
            "Added products to cart but no transaction was recorded"
        )
        actions.append(
            "Review checkout friction and consider a cart reminder"
        )

    # Rule 5: Little repeat activity
    if sessions <= 1 and active_days <= 2:
        reasons.append(
            "Limited repeat visits"
        )
        actions.append(
            "Test a relevant follow-up campaign"
        )

    # Fallback when no rule matches
    if not reasons:
        reasons.append(
            "No specific high-priority behavior rule matched"
        )
        actions.append(
            "Review the visitor journey before selecting an intervention"
        )

    return pd.Series({
        "root_cause_indicators": " | ".join(dict.fromkeys(reasons)),
        "recommended_actions": " | ".join(dict.fromkeys(actions))
    })


print("Analyzing visitor behavior...")

diagnoses = latest_df.apply(
    diagnose_visitor,
    axis=1
)

latest_df = pd.concat(
    [latest_df, diagnoses],
    axis=1
)

# --------------------------------------------------
# 5. Assign risk bands
# --------------------------------------------------

def assign_risk_band(score):
    if score >= 0.80:
        return "High"
    elif score >= 0.50:
        return "Medium"
    else:
        return "Low"


latest_df["risk_band"] = (
    latest_df["inactivity_risk_score"]
    .apply(assign_risk_band)
)

# --------------------------------------------------
# 6. Build report
# --------------------------------------------------

report_cols = [
    "visitorid",
    "snapshot_date",
    "inactivity_risk_score",
    "risk_band",
    "predicted_inactive",
    "days_since_last_activity",
    "days_since_first_activity",
    "session_count",
    "active_days",
    "view_events",
    "cart_events",
    "transaction_events",
    "root_cause_indicators",
    "recommended_actions"
]

report = latest_df[report_cols].copy()

report = report.sort_values(
    "inactivity_risk_score",
    ascending=False
)

report.to_csv(
    OUTPUT_PATH,
    index=False
)

# --------------------------------------------------
# 7. Display summary
# --------------------------------------------------

print("\n========== RISK BAND SUMMARY ==========")

print(
    report["risk_band"]
    .value_counts()
    .to_string()
)

print("\n========== TOP 10 HIGH-RISK VISITORS ==========")

print(
    report[
        [
            "visitorid",
            "inactivity_risk_score",
            "root_cause_indicators",
            "recommended_actions"
        ]
    ].head(10).to_string(index=False)
)

print("\nReport saved to:", OUTPUT_PATH)
print("\nRoot-cause diagnosis completed successfully!")
