# src/preprocessing.py

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        df["Churn"] = df["Churn"].replace({
            "Yes": 1,
            "No": 0
        })
        df["Churn"] = df["Churn"].astype(int)

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

    existing_service_columns = [
        col for col in service_columns if col in df.columns
    ]

    df["ServiceCount"] = 0

    for col in existing_service_columns:
        df["ServiceCount"] += (df[col].astype(str) == "Yes").astype(int)

    if "TechSupport" in df.columns:
        df["HasTechSupport"] = (
            df["TechSupport"].astype(str) == "Yes"
        ).astype(int)

    if "OnlineSecurity" in df.columns:
        df["HasOnlineSecurity"] = (
            df["OnlineSecurity"].astype(str) == "Yes"
        ).astype(int)

    if "PaymentMethod" in df.columns:
        df["AutoPayment"] = df["PaymentMethod"].isin([
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]).astype(int)

    if (
        "TotalCharges" in df.columns
        and "tenure" in df.columns
        and "MonthlyCharges" in df.columns
    ):
        safe_tenure = df["tenure"].replace(0, np.nan)
        df["AvgChargesPerMonth"] = df["TotalCharges"] / safe_tenure
        df["AvgChargesPerMonth"] = df["AvgChargesPerMonth"].fillna(
            df["MonthlyCharges"]
        )

    return df


def prepare_dataset(df: pd.DataFrame):
    df = clean_data(df)
    df = add_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    feature_columns = X.columns.tolist()

    return X, y, feature_columns, categorical_columns, numeric_columns


def create_preprocessor(numeric_columns, categorical_columns):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder)
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )

    return preprocessor


def prepare_input_data(
    df: pd.DataFrame,
    feature_columns,
    categorical_columns=None,
    numeric_columns=None
) -> pd.DataFrame:
    df = clean_data(df)
    df = add_features(df)

    categorical_columns = categorical_columns or []
    numeric_columns = numeric_columns or []

    for col in feature_columns:
        if col not in df.columns:
            if col in categorical_columns:
                df[col] = "Unknown"
            elif col in numeric_columns:
                df[col] = 0
            else:
                df[col] = 0

    df = df[feature_columns]

    return df
