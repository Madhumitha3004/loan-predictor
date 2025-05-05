import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and feature importance
model = joblib.load('loan_model.pkl')
fi_df = pd.read_csv('feature_importance.csv')

st.set_page_config(page_title="Loan Predictor", layout="wide")
st.title(" Loan Approval Predictor")
st.write("Provide your details below:")

# Collect user input
name = st.text_input("Name")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", 0)
coapplicant_income = st.number_input("Coapplicant Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", 0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_df = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Predict on button click
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Loan Status:")
    st.success(" Approved" if prediction == 1 else " Rejected")
    st.write(f"**Confidence Score:** {probability:.2f}")

    # Visualization: Feature Importance
    st.subheader(" Top Factors Influencing Decisions")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', palette='viridis', ax=ax1)
    ax1.set_title("Top 10 Important Features")
    st.pyplot(fig1)

    # Pie Chart Comparison Section
    st.subheader(" Your Inputs Compared with Approved vs Rejected")

    if os.path.exists('data/train.csv'):
        data = pd.read_csv('data/train.csv')
        data['Loan_Status'] = data['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected'})

        # Define simple buckets for comparison
        def bucket_income(val):
            if val < 2500: return 'Low'
            elif val < 5000: return 'Medium'
            else: return 'High'

        def bucket_loan(val):
            if val < 100: return 'Small'
            elif val < 200: return 'Medium'
            else: return 'Large'

        # Apply bucketing
        data['Income_Bucket'] = data['ApplicantIncome'].apply(bucket_income)
        data['Loan_Bucket'] = data['LoanAmount'].fillna(0).apply(bucket_loan)

        # Get your buckets
        user_income_bucket = bucket_income(applicant_income)
        user_loan_bucket = bucket_loan(loan_amount)

        st.markdown(f"**Your Income Bucket:** `{user_income_bucket}`")
        st.markdown(f"**Your Loan Amount Bucket:** `{user_loan_bucket}`")

        # Pie chart for income bucket
        fig2, ax2 = plt.subplots()
        income_counts = data[data['Income_Bucket'] == user_income_bucket]['Loan_Status'].value_counts()
        ax2.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"Approval Ratio for '{user_income_bucket}' Income")
        st.pyplot(fig2)

        # Pie chart for loan amount bucket
        fig3, ax3 = plt.subplots()
        loan_counts = data[data['Loan_Bucket'] == user_loan_bucket]['Loan_Status'].value_counts()
        ax3.pie(loan_counts, labels=loan_counts.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f"Approval Ratio for '{user_loan_bucket}' Loan Amount")
        st.pyplot(fig3)

    else:
        st.warning("⚠️ Could not load training data for comparison visualizations.")
