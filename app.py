import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgboost_heloc_model.pkl")

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

# Collect user input for all 23 features
st.subheader("📊 Applicant Financial Details")
col1, col2, col3 = st.columns(3)

with col1:
    ExternalRiskEstimate = st.number_input("Consolidated Risk Estimate", min_value=0, max_value=100, value=70)
    MSinceOldestTradeOpen = st.number_input("Months Since Oldest Trade Open", min_value=0, max_value=900, value=200)
    MSinceMostRecentTradeOpen = st.number_input("Months Since Most Recent Trade Open", min_value=0, max_value=400, value=10)
    AverageMInFile = st.number_input("Average Months in File", min_value=0, max_value=400, value=80)
    NumSatisfactoryTrades = st.number_input("Number of Satisfactory Trades", min_value=0, max_value=100, value=20)
    NumTrades60Ever2DerogPubRec = st.number_input("Number of Trades 60+ Ever Delinquent", min_value=0, max_value=50, value=1)
    NumTrades90Ever2DerogPubRec = st.number_input("Number of Trades 90+ Ever Delinquent", min_value=0, max_value=50, value=0)
    PercentTradesNeverDelq = st.number_input("Percent of Trades Never Delinquent", min_value=0, max_value=100, value=90)

with col2:
    MSinceMostRecentDelq = st.number_input("Months Since Most Recent Delinquency", min_value=0, max_value=100, value=5)
    MaxDelq2PublicRecLast12M = st.number_input("Max Delinquency/Public Records (Last 12M)", min_value=0, max_value=10, value=1)
    MaxDelqEver = st.number_input("Max Delinquency Ever", min_value=0, max_value=10, value=3)
    NumTotalTrades = st.number_input("Total Number of Trades", min_value=0, max_value=100, value=30)
    NumTradesOpeninLast12M = st.number_input("Trades Open in Last 12 Months", min_value=0, max_value=50, value=3)
    PercentInstallTrades = st.number_input("Percent Installment Trades", min_value=0, max_value=100, value=40)
    MSinceMostRecentInqexcl7days = st.number_input("Months Since Most Recent Inquiry (excl. 7 days)", min_value=0, max_value=50, value=2)
    NumInqLast6M = st.number_input("Number of Inquiries in Last 6 Months", min_value=0, max_value=20, value=2)

with col3:
    NumInqLast6Mexcl7days = st.number_input("Number of Inquiries (Last 6M, excl. 7 days)", min_value=0, max_value=20, value=2)
    NetFractionRevolvingBurden = st.number_input("Net Fraction Revolving Burden", min_value=0, max_value=300, value=30)
    NetFractionInstallBurden = st.number_input("Net Fraction Installment Burden", min_value=0, max_value=500, value=50)
    NumRevolvingTradesWBalance = st.number_input("Number of Revolving Trades with Balance", min_value=0, max_value=50, value=5)
    NumInstallTradesWBalance = st.number_input("Number of Installment Trades with Balance", min_value=0, max_value=50, value=2)
    NumBank2NatlTradesWHighUtilization = st.number_input("Bank/National Trades with High Utilization", min_value=0, max_value=20, value=1)
    PercentTradesWBalance = st.number_input("Percent of Trades with Balance", min_value=0, max_value=100, value=75)

# Create DataFrame from user inputs
input_features = [
    ExternalRiskEstimate, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen, AverageMInFile,
    NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq,
    MSinceMostRecentDelq, MaxDelq2PublicRecLast12M, MaxDelqEver, NumTotalTrades, NumTradesOpeninLast12M,
    PercentInstallTrades, MSinceMostRecentInqexcl7days, NumInqLast6M, NumInqLast6Mexcl7days,
    NetFractionRevolvingBurden, NetFractionInstallBurden, NumRevolvingTradesWBalance,
    NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization, PercentTradesWBalance
]

applicant_data = pd.DataFrame([input_features], columns=model.feature_names_in_)

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

