import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from model import train_model, prepare_input_row, predict_single
from visualize import plot_feature_summary

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")
st.title("👥 Employee Attrition Prediction")

uploaded_file = st.file_uploader("Upload employeeattri.csv", type="csv")


#when file is uploaded

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully!")

    #preprocessing
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data(df)

    #model train
    model, accuracy = train_model(X_train, y_train, X_test, y_test)

    st.subheader("🎯 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Test Samples", len(X_test))

    
    # INTERACTIVE INDIVIDUAL PREDICTION
    st.subheader("🔮 Predict Attrition for a Specific Employee")
    
    #form for input vlaues
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 60, 35) 
            daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 6000)

        with col2:
            distance_from_home = st.slider("Distance From Home", 1, 40, 10)
            years_at_company = st.slider("Years At Company", 0, 40, 5)
            years_promo = st.slider("Years Since Promotion", 0, 20, 2)

        with col3:
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4, 5], index=2)
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4, 5], index=2)
            overtime = st.selectbox("OverTime", ["No", "Yes"])

        submit = st.form_submit_button("Predict")

    if submit:
        #Converts the user inputs into a single-row dataframe 
        input_df = prepare_input_row(
            X_train.columns,
            age, daily_rate, monthly_income,
            distance_from_home, years_at_company, years_promo,
            job_satisfaction, work_life_balance, overtime
        )

        pred_label, pred_prob = predict_single(model, input_df)
        #pred_label -1(leaving) -0(staying)
        #pred_prob - probability of leaving
        
        st.subheader("Prediction Result")
        if pred_label == 1:
            st.error(f"🚨 HIGH RISK — Employee likely to leave ({pred_prob:.1%})")
        else:
            st.success(f"✅ LOW RISK — Employee likely to stay ({pred_prob:.1%})")

        st.progress(int(pred_prob * 100))

    # important feature
    st.subheader("📊 Top Factors Influencing Attrition")
    fig = plot_feature_summary(model, feature_cols)
    st.pyplot(fig)

else:
    st.info("👆 Upload your employeeattri.csv file to begin")