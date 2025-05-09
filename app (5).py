import streamlit as st
import joblib
import pandas as pd
import numpy as np


st.set_page_config(page_title="Heart Failure Predictor", layout="centered")

st.title("💓 Heart Failure Prediction App")
st.write("This app uses patient medical data to predict the risk of heart failure.")


@st.cache_resource

def load_model():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_model()

st.sidebar.header("Enter Patient Data")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 50)
    anaemia = st.sidebar.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.sidebar.slider("Creatinine Phosphokinase", 20, 8000, 250)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.sidebar.slider("Ejection Fraction", 10, 80, 38)
    high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0, 1])
    platelets = st.sidebar.slider("Platelets", 25000, 900000, 265000)
    serum_creatinine = st.sidebar.slider("Serum Creatinine", 0.1, 10.0, 1.2)
    serum_sodium = st.sidebar.slider("Serum Sodium", 110, 150, 137)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    smoking = st.sidebar.selectbox("Smoking", [0, 1])
    time = st.sidebar.slider("Follow-up Period (Days)", 0, 300, 130)

    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

st.subheader("Prediction Result")
st.write("🔍 Prediction: **{}**".format("Death" if prediction == 1 else "Survival"))
st.write("📊 Confidence: {:.2f}%".format(100 * np.max(proba)))

st.subheader("Input Patient Data")
st.dataframe(input_df)
