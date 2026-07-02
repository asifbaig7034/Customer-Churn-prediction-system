# src/predict.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import prepare_input_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


def load_artifact():
    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommendation(probability):
    if probability >= 0.70:
        return "Offer retention discount, priority support call, or contract upgrade."
    elif probability >= 0.40:
        return "Send loyalty offer, check service satisfaction, and promote annual contract."
    else:
        return "Maintain regular engagement and customer satisfaction tracking."


def predict_dataframe(df):
    artifact = load_artifact()

    model = artifact["model"]
    model_name = artifact["model_name"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]
    categorical_columns = artifact["categorical_columns"]
    numeric_columns = artifact["numeric_columns"]

    prepared_df = prepare_input_data(
        df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns
    )

    probabilities = model.predict_proba(prepared_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result_df = df.copy()
    result_df["churn_probability"] = probabilities
    result_df["prediction"] = np.where(predictions == 1, "Churn", "No Churn")
    result_df["risk_level"] = result_df["churn_probability"].apply(get_risk_level)
    result_df["recommendation"] = result_df["churn_probability"].apply(get_recommendation)
    result_df["model_name"] = model_name
    result_df["threshold"] = threshold

    return result_df


def predict_single_customer(customer_data):
    input_df = pd.DataFrame([customer_data])
    result_df = predict_dataframe(input_df)

    row = result_df.iloc[0]

    return {
        "churn_probability": round(float(row["churn_probability"]), 4),
        "prediction": str(row["prediction"]),
        "risk_level": str(row["risk_level"]),
        "recommendation": str(row["recommendation"]),
        "model_name": str(row["model_name"]),
        "threshold": float(row["threshold"])
    }


def predict_csv(input_csv_path, output_csv_path=None):
    input_csv_path = Path(input_csv_path)

    df = pd.read_csv(input_csv_path)
    result_df = predict_dataframe(df)

    if output_csv_path is None:
        output_csv_path = input_csv_path.parent / "churn_predictions.csv"

    output_csv_path = Path(output_csv_path)
    result_df.to_csv(output_csv_path, index=False)

    return result_df, output_csv_path


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.5,
        "TotalCharges": 452.5
    }

    prediction = predict_single_customer(sample_customer)

    print(prediction)
