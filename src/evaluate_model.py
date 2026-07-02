# src/evaluate_model.py

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from preprocessing import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
IMAGE_DIR = PROJECT_ROOT / "images"

IMAGE_DIR.mkdir(exist_ok=True)


def evaluate_saved_model():
    df_raw = pd.read_csv(DATA_PATH)

    X, y, feature_columns, categorical_columns, numeric_columns = prepare_dataset(df_raw)

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    model_name = artifact["model_name"]
    threshold = artifact["threshold"]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Model Name:", model_name)
    print("Threshold:", threshold)

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    cm = confusion_matrix(y, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix_full_data.png", dpi=300)
    plt.show()

    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curve_full_data.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    evaluate_saved_model()
