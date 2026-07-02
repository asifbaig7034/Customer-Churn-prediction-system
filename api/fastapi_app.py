
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"


app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI backend for predicting customer churn using a trained ML model.",
    version="1.0.0"
)


class CustomerInput(BaseModel):
    gender: str = Field(default="Female")
    SeniorCitizen: str = Field(default="No")
    Partner: str = Field(default="No")
    Dependents: str = Field(default="No")
    tenure: int = Field(default=12, ge=0)

    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="No")
    StreamingMovies: str = Field(default="No")

    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")

    MonthlyCharges: float = Field(default=70.0, ge=0)
    TotalCharges: Optional[float] = Field(default=None)


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    risk_level: str
    recommendation: str
    model_name: str
    threshold: float


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please train and save the model first."
        )

    return joblib.load(MODEL_PATH)


artifact = load_model_artifact()

model = artifact["model"]
model_name = artifact["model_name"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        median_total_charges = df["TotalCharges"].median()

        if pd.isna(median_total_charges):
            median_total_charges = 0

        df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({
            0: "No",
            1: "Yes",
            "0": "No",
            "1": "Yes"
        })

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72, 10000],
            labels=["0-1 year", "1-2 years", "2-4 years", "4-6 years", "6+ years"]
        )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    existing_service_columns = [col for col in service_columns if col in df.columns]

    df["ServiceCount"] = 0

    for col in existing_service_columns:
        df["ServiceCount"] += (df[col].astype(str) == "Yes").astype(int)

    if "TechSupport" in df.columns:
        df["HasTechSupport"] = (df["TechSupport"].astype(str) == "Yes").astype(int)

    if "OnlineSecurity" in df.columns:
        df["HasOnlineSecurity"] = (df["OnlineSecurity"].astype(str) == "Yes").astype(int)

    if "PaymentMethod" in df.columns:
        df["AutoPayment"] = df["PaymentMethod"].isin([
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]).astype(int)

    if "TotalCharges" in df.columns and "tenure" in df.columns and "MonthlyCharges" in df.columns:
        safe_tenure = df["tenure"].replace(0, np.nan)
        df["AvgChargesPerMonth"] = df["TotalCharges"] / safe_tenure
        df["AvgChargesPerMonth"] = df["AvgChargesPerMonth"].fillna(df["MonthlyCharges"])

    return df


def prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = add_features(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommendation(probability: float) -> str:
    if probability >= 0.70:
        return "Offer retention discount, priority support call, or contract upgrade."
    elif probability >= 0.40:
        return "Send loyalty offer, check service satisfaction, and promote annual contract."
    else:
        return "Maintain regular engagement and customer satisfaction tracking."


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = prepare_input_data(df)

    probabilities = model.predict_proba(prepared_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result_df = df.copy()
    result_df["churn_probability"] = probabilities
    result_df["prediction"] = np.where(predictions == 1, "Churn", "No Churn")
    result_df["risk_level"] = result_df["churn_probability"].apply(get_risk_level)
    result_df["recommendation"] = result_df["churn_probability"].apply(get_recommendation)

    return result_df


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "model_name": model_name
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_name,
        "threshold": threshold,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single_customer(customer: CustomerInput):
    try:
        customer_dict = customer.model_dump()

        if customer_dict["TotalCharges"] is None:
            customer_dict["TotalCharges"] = customer_dict["MonthlyCharges"] * max(customer_dict["tenure"], 1)

        input_df = pd.DataFrame([customer_dict])
        result_df = predict_dataframe(input_df)

        row = result_df.iloc[0]

        return {
            "churn_probability": round(float(row["churn_probability"]), 4),
            "prediction": str(row["prediction"]),
            "risk_level": str(row["risk_level"]),
            "recommendation": str(row["recommendation"]),
            "model_name": model_name,
            "threshold": float(threshold)
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict-batch")
def predict_batch_customers(customers: List[CustomerInput]):
    try:
        customer_records = []

        for customer in customers:
            customer_dict = customer.model_dump()

            if customer_dict["TotalCharges"] is None:
                customer_dict["TotalCharges"] = customer_dict["MonthlyCharges"] * max(customer_dict["tenure"], 1)

            customer_records.append(customer_dict)

        input_df = pd.DataFrame(customer_records)
        result_df = predict_dataframe(input_df)

        return {
            "total_customers": len(result_df),
            "model_name": model_name,
            "threshold": float(threshold),
            "predictions": result_df.to_dict(orient="records")
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")

        input_df = pd.read_csv(file.file)
        result_df = predict_dataframe(input_df)

        high_risk_count = int((result_df["risk_level"] == "High Risk").sum())
        medium_risk_count = int((result_df["risk_level"] == "Medium Risk").sum())
        low_risk_count = int((result_df["risk_level"] == "Low Risk").sum())

        return JSONResponse(
            content={
                "filename": file.filename,
                "total_customers": len(result_df),
                "high_risk_customers": high_risk_count,
                "medium_risk_customers": medium_risk_count,
                "low_risk_customers": low_risk_count,
                "model_name": model_name,
                "threshold": float(threshold),
                "predictions": result_df.to_dict(orient="records")
            }
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
