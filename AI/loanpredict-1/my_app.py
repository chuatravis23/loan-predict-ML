import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Utility functions
# =========================
def save_model(model, name):
    joblib.dump(model, f"{name}.joblib")

def load_model(name):
    file_path = f"{name}.joblib"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

# =========================
# Load dataset
# =========================
CSV_PATH = "loan_approval_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Encode target
df["loan_status"] = df["loan_status"].astype(str).str.strip().map({"Approved": 1, "Rejected": 0})

# Replace 0s in critical columns
critical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score']
for col in critical_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

# Fill missing values
for col in df.columns:
    if df[col].dtype.kind in "biufc":
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

# =========================
# App Header
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>💰 Loan Approval Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# =========================
# Tabs Navigation
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "🤖 Model Test Result", "📈 Visualization", "🔮 Loan Prediction"]
)

# =========================
# Dataset Overview
# =========================
with tab1:
    st.subheader("Dataset Overview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Sample Data")
        st.dataframe(df.head())
    with col2:
        st.write("### Class Distribution")
        fig = px.pie(df, names="loan_status", title="Loan Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.write("### Data Summary")
    st.write(df.describe())

# =========================
# Model Testing
# =========================
with tab2:
    st.subheader("Model Test Result")

    # One-hot encoding
    X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
    y = df['loan_status']
    feature_columns = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=150),
        "Support Vector Machine": SVC(kernel='rbf', probability=True)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for MSE/RMSE

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mse = mean_squared_error(y_test, y_prob)
        rmse = np.sqrt(mse)

        save_model(model, name)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "MSE": mse,
            "RMSE": rmse
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # --- Chart 1: Accuracy, Precision, Recall, F1 ---
    fig1 = px.bar(
        results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                        var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Performance (Accuracy, Precision, Recall, F1)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: MSE ---
    fig2 = px.bar(
        results_df, x="Model", y="MSE", color="Model",
        title="Mean Squared Error (MSE) Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Chart 3: RMSE ---
    fig3 = px.bar(
        results_df, x="Model", y="RMSE", color="Model",
        title="Root Mean Squared Error (RMSE) Comparison"
    )
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# Visualization
# =========================
with tab3:
    st.subheader("Visualization & Insights")

    # Correlation Heatmap (include loan_status)
    df_numeric = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df_numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar=True,
        ax=ax
    )
    ax.set_title("Correlation Matrix of Features (with Loan Status)")
    st.pyplot(fig)

    # Model Training (on full dataset)
    st.write("### Model Training Result")
    if os.path.exists("feature_columns.joblib") and os.path.exists("scaler.joblib"):
        feature_columns = joblib.load("feature_columns.joblib")
        scaler = joblib.load("scaler.joblib")

        X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X)
        y = df['loan_status']

        results = []
        for name in ["Logistic Regression", "Random Forest", "Support Vector Machine"]:
            model = load_model(name)
            if model:
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1]

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y, y_pred),
                    "Precision": precision_score(y, y_pred, zero_division=0),
                    "Recall": recall_score(y, y_pred, zero_division=0),
                    "F1 Score": f1_score(y, y_pred, zero_division=0),
                    "MSE": mean_squared_error(y, y_prob),
                    "RMSE": np.sqrt(mean_squared_error(y, y_prob))
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # --- Chart 1: Accuracy, Precision, Recall, F1 ---
        fig4 = px.bar(
            results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                            var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Full Dataset Performance (Accuracy, Precision, Recall, F1)"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # --- Chart 2: MSE ---
        fig5 = px.bar(
            results_df, x="Model", y="MSE", color="Model",
            title="Full Dataset Mean Squared Error (MSE)"
        )
        st.plotly_chart(fig5, use_container_width=True)

        # --- Chart 3: RMSE ---
        fig6 = px.bar(
            results_df, x="Model", y="RMSE", color="Model",
            title="Full Dataset Root Mean Squared Error (RMSE)"
        )
        st.plotly_chart(fig6, use_container_width=True)

# =========================
# Loan Prediction (with form)
# =========================
with tab4:
    st.subheader("Loan Approval Prediction")

    feature_columns = joblib.load("feature_columns.joblib")
    scaler = joblib.load("scaler.joblib")
    model = load_model("Random Forest")

    # Initialize session state for saving predictions
    if "prediction_data" not in st.session_state:
        st.session_state["prediction_data"] = pd.DataFrame(columns=[
            "No. of Dependents", "Annual Income", "Loan Amount", "Loan Term", 
            "CIBIL Score", "Residential Assets", "Commercial Assets", 
            "Luxury Assets", "Bank Assets", "Education", "Self Employed", 
            "Prediction"
        ])

    # Show validation rules for user
    st.info("""
    ⚠️ **Validation Rules & Limitations**:
    - Annual Income must be **at least ₹640,000 (INR)**  
    - Loan Term must be **at least 2 year and maximum 20 year**  
    - CIBIL Score must be **between 300 and 900**
    """)

    # Form to prevent auto-refresh
    with st.form("loan_form"):
        raw = {}
        st.write("### Enter Loan Applicant Information")

        # Input fields with explanations
        c1, c2 = st.columns([2, 3])
        with c1:
            raw['no_of_dependents'] = st.number_input("No. of Dependents", min_value=0.0, step=1.0)
        with c2:
            st.caption("👨‍👩‍👧 Number of people financially dependent on the applicant.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['income_annum'] = st.number_input("Annual Income (₹ INR)", min_value=0.0, step=100000.0, format="%.0f")
        with c2:
            st.caption("💵 Total yearly income of the applicant in **Indian Rupee (₹)**.")

        c1, c2 = st.columns([2, 3])
        with c1:
            max_loan_amount = raw.get("income_annum", 0) * 5 if raw.get("income_annum", 0) > 0 else None
            raw['loan_amount'] = st.number_input(
                "Loan Amount (₹ INR)", 
                min_value=100000.0, 
                max_value=max_loan_amount if max_loan_amount else 100000000.0, 
                step=100000.0, 
                format="%.0f"
            )
        with c2:
            st.caption("🏦 Requested loan (must be **≥ ₹100,000**).")
        with c1:
            raw['loan_term'] = st.number_input("Loan Term (2-20years)", min_value=0.0, step=1.0, format="%.0f")
        with c2:
            st.caption("📅 Duration of the loan in years.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['cibil_score'] = st.number_input("CIBIL Score", min_value=0.0, step=1.0, format="%.0f")
        with c2:
            st.caption("⭐ Credit score between 300–900. Higher = more creditworthy.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['residential_assets_value'] = st.number_input("Residential Assets Value (₹ INR)", min_value=0.0, step=1.0)
        with c2:
            st.caption("🏠 Value of houses, apartments, or other residential properties in **₹ INR**.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['commercial_assets_value'] = st.number_input("Commercial Assets Value (₹ INR)", min_value=0.0, step=1.0)
        with c2:
            st.caption("🏢 Value of shops, offices, or other commercial properties in **₹ INR**.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['luxury_assets_value'] = st.number_input("Luxury Assets Value (₹ INR)", min_value=0.0, step=1.0)
        with c2:
            st.caption("💎 Value of luxury items such as cars, jewelry, or high-value assets in **₹ INR**.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['bank_asset_value'] = st.number_input("Bank Asset Value (₹ INR)", min_value=0.0, step=1.0)
        with c2:
            st.caption("🏦 Total deposits, savings, and investments in **₹ INR**.")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['education'] = st.selectbox("Education", df['education'].unique())
        with c2:
            st.caption("🎓 Education level of the applicant (e.g., Graduate, Non-Graduate).")

        c1, c2 = st.columns([2, 3])
        with c1:
            raw['self_employed'] = st.selectbox("Self Employed", df['self_employed'].unique())
        with c2:
            st.caption("💼 Whether the applicant is self-employed or salaried.")

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Validation
        if raw.get("income_annum", 0) < 640000:
            st.error("❌ Prediction aborted: Annual income must be at least ₹640,000.")
        elif raw.get("loan_term", 0) < 1:
            st.error("❌ Prediction aborted: Loan term must be at least 1 year.")
        elif raw.get("cibil_score", 0) < 300 or raw.get("cibil_score", 0) > 900:
            st.error("❌ Prediction aborted: CIBIL Score must be between 300 and 900.")
        elif raw.get("loan_term", 0) < 2 or raw.get("loan_term", 0) > 20:
            st.error("❌ Prediction aborted: Loan term must be between 2 and 20 years.")
        elif raw.get("loan_amount", 0) < 100000:
            st.error("❌ Prediction aborted: Loan amount must be at least ₹100,000.")
        else:
            new_df = pd.DataFrame([raw])
            new_df_dum = pd.get_dummies(new_df, drop_first=True).reindex(columns=feature_columns, fill_value=0)
            new_df_scaled = scaler.transform(new_df_dum)

            prediction = model.predict(new_df_scaled)[0]

            # Show result
            if prediction == 1:
                st.success("✅ Loan Approved")
                pred_label = "Approved"
            else:
                st.error("❌ Loan Rejected")
                pred_label = "Rejected"

            # Save prediction into session_state dataset
            new_row = {
                "No. of Dependents": raw['no_of_dependents'],
                "Annual Income": raw['income_annum'],
                "Loan Amount": raw['loan_amount'],
                "Loan Term": raw['loan_term'],
                "CIBIL Score": raw['cibil_score'],
                "Residential Assets": raw['residential_assets_value'],
                "Commercial Assets": raw['commercial_assets_value'],
                "Luxury Assets": raw['luxury_assets_value'],
                "Bank Assets": raw['bank_asset_value'],
                "Education": raw['education'],
                "Self Employed": raw['self_employed'],
                "Prediction": pred_label
            }
            st.session_state["prediction_data"] = pd.concat(
                [st.session_state["prediction_data"], pd.DataFrame([new_row])],
                ignore_index=True
            )
    # Show prediction history
    if len(st.session_state["prediction_data"]) > 0:
        st.subheader("📊 Prediction History")
        st.dataframe(st.session_state["prediction_data"])

        # Download button
        csv = st.session_state["prediction_data"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    
