
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
DATA_PATH = (
    ROOT / "data" / "processed" / "churn_training_data.csv.gz"
)

OUTPUT_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading churn dataset and trained model...")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="raise"
)

df = df.sort_values(
    ["snapshot_date", "visitorid"]
).reset_index(drop=True)

target = "churn_label"

excluded = {
    "visitorid",
    "snapshot_date",
    target
}

feature_cols = [
    col for col in df.columns
    if col not in excluded
]

# --------------------------------------------------
# 3. Recreate the chronological test period
# --------------------------------------------------

dates = sorted(df["snapshot_date"].unique())

if len(dates) < 2:
    raise ValueError("At least two snapshot dates are required.")

split_index = int(len(dates) * 0.8)
split_index = max(1, min(split_index, len(dates) - 1))

test_start_date = dates[split_index]

test_df = df[
    df["snapshot_date"] >= test_start_date
].copy()

print("Test records:", f"{len(test_df):,}")
print("Test period starts:", pd.Timestamp(test_start_date).date())

# --------------------------------------------------
# 4. Load trained Random Forest pipeline
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

classifier = pipeline.named_steps["model"]

# --------------------------------------------------
# 5. Calculate built-in feature importance
# --------------------------------------------------

print("\nCalculating Random Forest feature importance...")

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": classifier.feature_importances_
}).sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance["importance_percent"] = (
    importance["importance"] * 100
).round(2)

# Save feature importance
importance_path = (
    OUTPUT_DIR / "feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)

# --------------------------------------------------
# 6. Display top 10 features
# --------------------------------------------------

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(
    importance[
        ["feature", "importance_percent"]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# 7. Generate feature importance plot
# --------------------------------------------------

top_features = importance.head(15).sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Behavioral Features")
plt.title("Random Forest - Top 15 Feature Importances")

plt.tight_layout()

plot_path = (
    FIGURE_DIR / "feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. Completion summary
# --------------------------------------------------

print("\n========== EXPLANATION COMPLETE ==========")
print("Feature importance CSV:", importance_path)
print("Feature importance plot:", plot_path)

print(
    "\nNote: Feature importance indicates how much "
    "the Random Forest relies on each feature. "
    "It does not show whether a feature increases "
    "or decreases inactivity risk."
)

print("\nFeature importance analysis completed successfully!")
