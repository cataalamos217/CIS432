import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time  # Add this for simulating the delay

# Load the trained model
model = joblib.load("xgboost_heloc_model.pkl")

# Streamlit App Title
st.set_page_config(page_title="HELOC Eligibility Screening", layout="wide")

# Add a smaller left-aligned header with "Simon Bank of Rochester" in light grey
st.markdown("<h3 style='text-align: left; color: lightgrey; font-size: 14px;'>Simon Bank of Rochester</h3>", unsafe_allow_html=True)

# Existing title and description
st.markdown("<h2 style='text-align: left;'>Home Equity Line of Credit (HELOC) Eligibility Screening</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>Thank you for providing your financial details. This information, which can typically be found in your credit report, will help us assess your eligibility for a Home Equity Line of Credit (HELOC). Your details will be securely processed, and we’ll notify you of the outcome. If your application is not eligible, we’ll provide suggestions on how to improve your chances in future applications.</p>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: left;'>Let's get started with your application.</h4>", unsafe_allow_html=True)
st.write("---")

# Apply custom CSS to limit input field width
st.markdown("""
    <style>
    div[data-baseweb="input"] {
        max-width: 300px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Collect user input for all 8 features in a single column with shorter fields
with st.container(): 
    ExternalRiskEstimate = st.number_input("""Please provide your External Risk Estimate. This should be a value between 0 and 100.  
_This can typically be found on your credit report. External risk estimate is a consolidated credit risk score based on factors like credit history and payment behavior._
""", min_value=0, max_value=100)

AverageMInFile = st.number_input("""
Please provide the average number of months your credit accounts have been open.  
_This information can typically be found on your credit report under 'Average Age of Accounts' or a similar section._
""", min_value=0, max_value=400)

NumSatisfactoryTrades = st.number_input("""
Please provide the number of credit accounts you currently have in good standing (no late payments).  
_This includes accounts that are not overdue or in default._
""", min_value=0, max_value=75)

PercentTradesNeverDelq = st.number_input("""
Please provide the percentage of your credit accounts that have never been delinquent. Your input should be expressed as a value between 0 and 100.  
_Max value = 100. This includes accounts that have never had a late payment._
""", min_value=0, max_value=100)

# Display the delinquency status scale
st.markdown("""
- **0** = I have a derogatory comment on my credit history (e.g., collections, charge-offs).  
- **1** = I have been 120+ days delinquent on a payment.  
- **2** = I have been 90 days delinquent on a payment.  
- **3** = I have been 60 days delinquent on a payment.  
- **4** = I have been 30 days delinquent on a payment.  
- **5** = I am unsure about the exact delinquency status, but some delinquency occurred.  
- **7** = I have never had a delinquency reported on my credit history.  
- **8** = I have a special case, such as bankruptcy or foreclosure, that affects my credit history.  
""", unsafe_allow_html=True)
  


# Number input for delinquency status with max_value 8
MaxDelq2PublicRecLast12M = st.number_input("""
Using one of the above values, please provide the worst delinquency status recorded in the last 12 months.  
_Enter a number corresponding to the options above._
""", min_value=0, max_value=8, value = 7)

MSinceMostRecentInqexcl7days = st.number_input("""
Please provide the number of months since your most recent credit inquiry, excluding the last 7 days.  
_This helps exclude any recent activity that may not yet be fully reflected in your credit report._
""", min_value=0, max_value=50)

NetFractionRevolvingBurden = st.number_input("""
Please provide the proportion of your revolving credit used relative to your credit limit.  
_This is the percentage of your credit limit that is currently being utilized._
""", min_value=0, max_value=300)

NumBank2NatlTradesWHighUtilization = st.number_input("""
Please provide the number of bank-issued credit accounts with a high balance relative to the credit limit.  
_This refers to accounts where the balance is over 30% of the credit limit._
""", min_value=0, max_value=20)

# Create DataFrame from user inputs
input_features = [
        ExternalRiskEstimate,
        NetFractionRevolvingBurden,
        PercentTradesNeverDelq,
        MSinceMostRecentInqexcl7days,
        AverageMInFile,
        MaxDelq2PublicRecLast12M,
        NumSatisfactoryTrades,
        NumBank2NatlTradesWHighUtilization
]

applicant_data = pd.DataFrame([input_features], columns=model.feature_names_in_)

# Create a submit button
submit_button = st.button("Submit my application")

# Check Eligibility
if submit_button:
    with st.spinner('Checking eligibility... This may take a few seconds.'):
        time.sleep(2)  # Simulate a small delay (2 seconds)
        prediction = model.predict(applicant_data)[0]
        probability = model.predict_proba(applicant_data)[0][1]  # Probability of being "Bad" (denied)

    st.write("---")
    if prediction == 1:
        st.markdown("### **Unfortunately, your application does not meet the current eligibility criteria.**")
        st.warning(f"Based on your financial details, your application does not meet the initial eligibility criteria. "
                   f"Probability of denial: **{probability:.2%}**")
        st.info("**Note:** Improving your credit score, reducing debt, or increasing trade history may enhance your chances for future applications.")
    else:
        st.markdown("### **We are glad to inform you your application is eligible for further review by our loan officers**")
        st.success(f"Probability of Approval: **{(1 - probability):.2%}**")
        st.info("📌 **Note:** Final approval depends on additional verification by Simon Bank.")

st.write("---")
st.markdown(" *This tool is a working prototype. It is meant to provide an initial assessment. Your information may be sent to loan officers for manual review during this time.*")
