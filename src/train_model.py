# src/train_model.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import prepare_dataset, create_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

MODEL_DIR.mkdir(exist_ok=True)


def evaluate_model(model_name, model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    }


def tune_threshold(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold_results = []

    for threshold in np.arange(0.20, 0.81, 0.05):
        y_pred = (y_proba >= threshold).astype(int)

        threshold_results.append({
            "Threshold": round(float(threshold), 2),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })

    threshold_df = pd.DataFrame(threshold_results)

    best_threshold_row = threshold_df.sort_values(
        by=["F1 Score", "Recall"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best_threshold_row["Threshold"])

    return best_threshold, threshold_df


def train():
    df_raw = pd.read_csv(DATA_PATH)

    X, y, feature_columns, categorical_columns, numeric_columns = prepare_dataset(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )
    }

    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    results = []

    for model_name, model in trained_models.items():
        result = evaluate_model(model_name, model, X_test, y_test)
        results.append(result)

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values(
        by=["F1 Score", "Recall", "ROC AUC"],
        ascending=False
    )

    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    best_threshold, threshold_df = tune_threshold(
        best_model,
        X_test,
        y_test
    )

    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns
    }

    joblib.dump(model_artifact, MODEL_PATH)

    metrics_df.to_csv(PROJECT_ROOT / "model_metrics.csv", index=False)
    threshold_df.to_csv(PROJECT_ROOT / "threshold_results.csv", index=False)

    print("\nTraining completed successfully.")
    print("Best model:", best_model_name)
    print("Best threshold:", best_threshold)
    print("Model saved at:", MODEL_PATH)
    print("\nModel metrics:")
    print(metrics_df)


if __name__ == "__main__":
    train()
