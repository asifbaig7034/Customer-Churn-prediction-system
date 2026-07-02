import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Predict whether a telecom customer is likely to leave the company.")

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

try:
    artifact = load_model_artifact()

    model = artifact["model"]
    model_name = artifact["model_name"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

except Exception as error:
    st.error("App failed while loading the trained model.")
    st.exception(error)
    st.stop()


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

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


def add_features(df):
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


def prepare_input_data(df):
    df = clean_data(df)
    df = add_features(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


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


st.sidebar.header("Model Information")
st.sidebar.write("Model used:", model_name)
st.sidebar.write("Decision threshold:", threshold)

tab1, tab2, tab3 = st.tabs([
    "Single Customer Prediction",
    "Batch Prediction",
    "Project Info"
])

with tab1:
    st.header("Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure in months", min_value=0, max_value=100, value=12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

    with col5:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(monthly_charges * max(tenure, 1))
        )

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    if st.button("Predict Churn"):
        prepared_input = prepare_input_data(input_df)

        probability = model.predict_proba(prepared_input)[:, 1][0]
        prediction = int(probability >= threshold)

        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(probability)

        col_result1, col_result2, col_result3 = st.columns(3)

        with col_result1:
            st.metric("Churn Probability", f"{probability:.2%}")

        with col_result2:
            st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

        with col_result3:
            st.metric("Risk Level", risk_level)

        if risk_level == "High Risk":
            st.error(recommendation)
        elif risk_level == "Medium Risk":
            st.warning(recommendation)
        else:
            st.success(recommendation)

        st.subheader("Input Customer Data")
        st.dataframe(input_df)

with tab2:
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing customer data",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(batch_df.head())

        prepared_batch = prepare_input_data(batch_df)

        probabilities = model.predict_proba(prepared_batch)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        result_df = batch_df.copy()
        result_df["Churn_Probability"] = probabilities
        result_df["Prediction"] = np.where(predictions == 1, "Churn", "No Churn")
        result_df["Risk_Level"] = result_df["Churn_Probability"].apply(get_risk_level)
        result_df["Recommendation"] = result_df["Churn_Probability"].apply(get_recommendation)

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Total Customers", len(result_df))

        with col_b:
            high_risk_count = (result_df["Risk_Level"] == "High Risk").sum()
            st.metric("High Risk Customers", int(high_risk_count))

        with col_c:
            average_probability = result_df["Churn_Probability"].mean()
            st.metric("Average Churn Probability", f"{average_probability:.2%}")

        st.subheader("Risk Level Distribution")

        risk_counts = result_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]

        fig = px.bar(
            risk_counts,
            x="Risk_Level",
            y="Count",
            title="Customer Risk Level Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

with tab3:
    st.header("Project Information")

    st.markdown("""
    ## Problem Statement

    Customer churn is a major problem for subscription-based companies.
    This project predicts whether a telecom customer is likely to leave the company.

    ## Project Workflow

    1. Data cleaning
    2. Exploratory data analysis
    3. Feature engineering
    4. Model training
    5. Model evaluation
    6. Threshold tuning
    7. Streamlit deployment

    ## Models Trained

    - Logistic Regression
    - Random Forest

    ## Business Use

    The company can use this system to identify high-risk customers and take retention actions such as:

    - support calls
    - loyalty offers
    - discounts
    - contract upgrades
    - service improvement campaigns
    """)
