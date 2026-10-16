
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "churn_training_data.csv.gz"
)

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

print("Loading churn dataset...")

df = pd.read_csv(DATA_PATH)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="raise"
)

df = df.sort_values(
    ["snapshot_date", "visitorid"]
).reset_index(drop=True)

# --------------------------------------------------
# 2. Recreate the same chronological test split
# --------------------------------------------------

unique_dates = sorted(df["snapshot_date"].unique())

if len(unique_dates) < 2:
    raise ValueError("At least two snapshot dates are required.")

split_index = int(len(unique_dates) * 0.8)
split_index = max(
    1,
    min(split_index, len(unique_dates) - 1)
)

test_start_date = unique_dates[split_index]

test_mask = df["snapshot_date"] >= test_start_date
test_df = df.loc[test_mask].copy()

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

X_test = test_df[feature_cols].copy()
y_test = test_df[target].astype(int)

print("\n========== TEST DATA ==========")
print("Test rows:", f"{len(test_df):,}")
print("Test starts:", test_start_date)
print("Actual label distribution:")
print(y_test.value_counts().sort_index())

# --------------------------------------------------
# 3. Majority-class baseline
# --------------------------------------------------

# This baseline always predicts the most common
# class in the training period.

train_df = df.loc[~test_mask]
majority_class = int(
    train_df[target].value_counts().idxmax()
)

baseline_predictions = [
    majority_class
] * len(y_test)

print("\n========== MAJORITY-CLASS BASELINE ==========")
print("Always predicts class:", majority_class)

print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Balanced accuracy:",
    round(
        balanced_accuracy_score(y_test, baseline_predictions),
        4
    )
)

print("\nBaseline confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))

# --------------------------------------------------
# 4. Evaluate saved models
# --------------------------------------------------

model_files = {
    "Logistic Regression":
        MODEL_DIR / "logistic_regression.joblib",

    "Random Forest":
        MODEL_DIR / "random_forest.joblib"
}

results = []

for model_name, model_path in model_files.items():

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print(f"\n========== {model_name.upper()} ==========")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Find the probability columns by their class labels.
    classes = list(model.named_steps["model"].classes_)

    class_0_index = classes.index(0)
    class_1_index = classes.index(1)

    probability_class_0 = probabilities[:, class_0_index]
    probability_class_1 = probabilities[:, class_1_index]

    # Class 0 is the minority class: visitors who return
    # during the next 14 days.
    y_class_0 = (y_test == 0).astype(int)

    minority_average_precision = average_precision_score(
        y_class_0,
        probability_class_0
    )

    # Also report AP for class 1 for completeness.
    y_class_1 = (y_test == 1).astype(int)

    class_1_average_precision = average_precision_score(
        y_class_1,
        probability_class_1
    )

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nAdditional evaluation metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(
        "Average precision - class 0 (returning): "
        f"{minority_average_precision:.4f}"
    )
    print(
        "Average precision - class 1 (inactive): "
        f"{class_1_average_precision:.4f}"
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "average_precision_class_0": minority_average_precision,
        "average_precision_class_1": class_1_average_precision
    })

# --------------------------------------------------
# 5. Save evaluation comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

comparison_path = (
    OUTPUT_DIR / "model_evaluation_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

print("\nSaved evaluation results to:")
print(comparison_path)

print("\nModel evaluation completed successfully!")
