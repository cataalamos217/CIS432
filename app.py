import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost model with 10 selected features
model = joblib.load("xgboost_heloc_selected.pkl")

# Define the 10 selected features
selected_features = [
    "ExternalRiskEstimate",
    "PercentTradesNeverDelq",
    "MSinceMostRecentInqexcl7days",
    "NetFractionRevolvingBurden",
    "NumBank2NatlTradesWHighUtilization",
    "MaxDelq2PublicRecLast12M",
    "AverageMInFile",
    "NumTrades60Ever2DerogPubRec",
    "MaxDelqEver",
    "PercentTradesWBalance"
]

# Streamlit App Title
st.set_page_config(page_title="HELOC Eligibility Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏡 HELOC Eligibility Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:gray;'>📌 Enter financial details below to check eligibility for a Home Equity Line of Credit (HELOC).</h4>", unsafe_allow_html=True)
st.write("---")

# Sidebar for user instructions
st.sidebar.header("🔍 About this Tool")
st.sidebar.write(
    "This tool provides an **initial assessment** of HELOC eligibility based on financial history. "
    "Final decisions are **subject to manual review** by loan officers."
)

# Collect user input for the selected 10 features
st.subheader("📊 Applicant Financial Details")
col1, col2 = st.columns(2)

with col1:
    ExternalRiskEstimate = st.number_input("Consolidated Risk Estimate", min_value=0, max_value=100, value=70)
    PercentTradesNeverDelq = st.number_input("Percent of Trades Never Delinquent", min_value=0, max_value=100, value=90)
    MSinceMostRecentInqexcl7days = st.number_input("Months Since Most Recent Inquiry (excl. 7 days)", min_value=0, max_value=50, value=2)
    NetFractionRevolvingBurden = st.number_input("Net Fraction Revolving Burden", min_value=0, max_value=300, value=30)
    NumBank2NatlTradesWHighUtilization = st.number_input("Bank/National Trades with High Utilization", min_value=0, max_value=20, value=1)

with col2:
    MaxDelq2PublicRecLast12M = st.number_input("Max Delinquency/Public Records (Last 12M)", min_value=0, max_value=10, value=1)
    AverageMInFile = st.number_input("Average Months in File", min_value=0, max_value=400, value=80)
    NumTrades60Ever2DerogPubRec = st.number_input("Number of Trades 60+ Ever Delinquent", min_value=0, max_value=50, value=1)
    MaxDelqEver = st.number_input("Max Delinquency Ever", min_value=0, max_value=10, value=3)
    PercentTradesWBalance = st.number_input("Percent of Trades with Balance", min_value=0, max_value=100, value=75)

# Create DataFrame from user inputs
input_features = [
    ExternalRiskEstimate, PercentTradesNeverDelq, MSinceMostRecentInqexcl7days,
    NetFractionRevolvingBurden, NumBank2NatlTradesWHighUtilization, MaxDelq2PublicRecLast12M,
    AverageMInFile, NumTrades60Ever2DerogPubRec, MaxDelqEver, PercentTradesWBalance
]

applicant_data = pd.DataFrame([input_features], columns=selected_features)

# Check Eligibility
if st.button("🔍 Check Eligibility"):
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

st.write("---")
st.markdown("📌 *This tool provides an initial assessment. Final decisions are subject to manual review.*")
