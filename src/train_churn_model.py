
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Paths and folders
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "churn_training_data.csv.gz"
)

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load churn dataset
# --------------------------------------------------

print("Loading churn dataset...")

df = pd.read_csv(DATA_PATH)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="raise"
)

# Keep ALL eligible snapshots.
# Do not deduplicate visitors.
# Features for each snapshot are based only on its
# historical observation window.

df = df.sort_values(
    ["snapshot_date", "visitorid"]
).reset_index(drop=True)

print("\n========== DATASET SUMMARY ==========")
print("Snapshot rows:", f"{len(df):,}")
print("Unique visitors:", f"{df['visitorid'].nunique():,}")
print("Number of snapshots:", df["snapshot_date"].nunique())

print("\nOverall label distribution:")
print(df["churn_label"].value_counts().sort_index())

print("\nOverall label percentages:")
print(
    (df["churn_label"].value_counts(normalize=True) * 100)
    .sort_index()
    .round(2)
)

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

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

X = df[feature_cols].copy()
y = df[target].astype(int)

# --------------------------------------------------
# 4. Chronological train-test split
# --------------------------------------------------

# Older snapshots are used for training.
# Newer snapshots are reserved for testing.

unique_dates = sorted(df["snapshot_date"].unique())

if len(unique_dates) < 2:
    raise ValueError(
        "Need at least two snapshot dates for a "
        "chronological train-test split."
    )

split_index = int(len(unique_dates) * 0.8)

split_index = max(
    1,
    min(split_index, len(unique_dates) - 1)
)

test_start_date = unique_dates[split_index]

train_mask = df["snapshot_date"] < test_start_date
test_mask = df["snapshot_date"] >= test_start_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# --------------------------------------------------
# 5. Split diagnostics
# --------------------------------------------------

print("\n========== SPLIT DIAGNOSTICS ==========")

print("\nTraining rows:", f"{len(X_train):,}")
print("Training labels:")
print(y_train.value_counts().sort_index())

print("\nTest rows:", f"{len(X_test):,}")
print("Test labels:")
print(y_test.value_counts().sort_index())

print("\nTraining label percentages:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .sort_index()
    .round(2)
)

print("\nTest label percentages:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .sort_index()
    .round(2)
)

print("\nTraining dates:")
print(
    df.loc[train_mask, "snapshot_date"].min(),
    "to",
    df.loc[train_mask, "snapshot_date"].max()
)

print("\nTest dates:")
print(
    df.loc[test_mask, "snapshot_date"].min(),
    "to",
    df.loc[test_mask, "snapshot_date"].max()
)

if X_train.empty or X_test.empty:
    raise ValueError(
        "Training or test dataset is empty. Check the split dates."
    )

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training or test data has only one class. "
        "Inspect the label distribution and split dates."
    )

print("\n========== TEMPORAL SPLIT ==========")
print("Training rows:", f"{len(X_train):,}")
print("Testing rows:", f"{len(X_test):,}")
print(
    "Test begins:",
    pd.Timestamp(test_start_date).date()
)
print("Training churn rate:", round(y_train.mean(), 4))
print("Testing churn rate:", round(y_test.mean(), 4))

# --------------------------------------------------
# 6. Preprocessing pipeline
# --------------------------------------------------

# All model features are numerical.
# Missing values are imputed with the training median.
# Scaling is included for Logistic Regression.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            feature_cols
        )
    ]
)

# --------------------------------------------------
# 7. Define machine learning models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 8. Train and evaluate models
# --------------------------------------------------

results = []

for name, classifier in models.items():

    print(f"\n========== {name.upper()} ==========")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    print("Training model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    results.append(metrics)

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

    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {metric_value:.4f}")

    # Save trained model
    model_name = name.lower().replace(" ", "_")

    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(pipeline, model_path)

    print("Saved model:", model_path)

# --------------------------------------------------
# 9. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# --------------------------------------------------
# 10. Final status
# --------------------------------------------------

print("\n========== TRAINING COMPLETE ==========")
print("Saved models in:", MODEL_DIR)
print("Saved comparison in:", results_path)
