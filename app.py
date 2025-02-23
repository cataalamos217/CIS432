import streamlit as st
import pandas as pd
import joblib

# ✅ Load the trained XGBoost model with 8 selected features
model_filename = "xgboost_heloc_refined.pkl"  # Make sure this matches your saved model
try:
    model = joblib.load(model_filename)
    expected_features = list(model.feature_names_in_)  # Extract feature names from model
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# ✅ Define the 8 selected features
selected_features = [
    "ExternalRiskEstimate",
    "PercentTradesNeverDelq",
    "MSinceMostRecentInqexcl7days",
    "NetFractionRevolvingBurden",
    "NumBank2NatlTradesWHighUtilization",
    "MaxDelq2PublicRecLast12M",
    "AverageMInFile",
    "MaxDelqEver"
]

# ✅ Check for feature mismatch
if set(selected_features) != set(expected_features):
    st.error("⚠️ Feature mismatch! The selected features do not match the model's expected input.")
    st.write(f"Expected: {expected_features}")
    st.write(f"Provided: {selected_features}")
    st.stop()

# 🎨 Streamlit UI
st.set_page_config(page_title="HELOC Eligibility Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏡 HELOC Eligibility Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:gray;'>📌 Enter financial details below to check eligibility for a Home Equity Line of Credit (HELOC).</h4>", unsafe_allow_html=True)
st.write("---")

# ℹ️ Sidebar Information
st.sidebar.header("🔍 About this Tool")
st.sidebar.write(
    "This tool provides an **initial assessment** of HELOC eligibility based on financial history. "
    "Final decisions are **subject to manual review** by loan officers."
)

# 📊 Collect user input for the selected 8 features
st.subheader("📊 Applicant Financial Details")
col1, col2 = st.columns(2)

with col1:
    ExternalRiskEstimate = st.number_input("Consolidated Risk Estimate", min_value=0, max_value=100, value=70)
    PercentTradesNeverDelq = st.number_input("Percent of Trades Never Delinquent", min_value=0, max_value=100, value=90)
    MSinceMostRecentInqexcl7days = st.number_input("Months Since Most Recent Inquiry (excl. 7 days)", min_value=0, max_value=50, value=2)
    NetFractionRevolvingBurden = st.number_input("Net Fraction Revolving Burden", min_value=0, max_value=300, value=30)

with col2:
    NumBank2NatlTradesWHighUtilization = st.number_input("Bank/National Trades with High Utilization", min_value=0, max_value=20, value=1)
    MaxDelq2PublicRecLast12M = st.number_input("Max Delinquency/Public Records (Last 12M)", min_value=0, max_value=10, value=1)
    AverageMInFile = st.number_input("Average Months in File", min_value=0, max_value=400, value=80)
    MaxDelqEver = st.number_input("Max Delinquency Ever", min_value=0, max_value=10, value=3)

# ✅ Create DataFrame from user inputs
input_features = [
    ExternalRiskEstimate, PercentTradesNeverDelq, MSinceMostRecentInqexcl7days,
    NetFractionRevolvingBurden, NumBank2NatlTradesWHighUtilization,
    MaxDelq2PublicRecLast12M, AverageMInFile, MaxDelqEver
]

applicant_data = pd.DataFrame([input_features], columns=selected_features)

# ✅ Check Eligibility
if st.button("🔍 Check Eligibility"):
    try:
        prediction = model.predict(applicant_data)[0]
        probability = model.predict_proba(applicant_data)[0][1]  # Probability of being "Bad" (denied)

        st.write("---")
        if prediction == 1:
            st.markdown("### ❌ **Application Denied**")
            st.warning(f"Unfortunately, your HELOC application does not meet initial eligibility criteria. "
                       f"Probability of Denial: **{probability:.2%}**")
            st.info("📌 **Advice:** Improving your credit score, reducing debt, or increasing trade history may improve future applications.")
        else:
            st.markdown("### ✅ **Application Sent for Review**")
            st.success(f"Your HELOC application is eligible for further review by loan officers. "
                       f"Probability of Approval: **{(1 - probability):.2%}**")
            st.info("📌 **Note:** Final approval depends on additional verification by Simon Bank.")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

st.write("---")
st.markdown("📌 *This tool provides an initial assessment. Final decisions are subject to manual review.*")
