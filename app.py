import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
from catboost import CatBoostClassifier

# -----------------------------
# Load trained CatBoost model
# -----------------------------
loaded_model = CatBoostClassifier()
loaded_model.load_model("diabetes_model.cbm")

# Load scaler used during training
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Helper functions
# -----------------------------
binary_map = {"Yes": 1, "No": 0}

def preprocess_input(raw_data):
    """Convert user input into numeric format (0/1 for Yes/No) in correct column order"""
    return {
        "HighBP": binary_map[raw_data["HighBP"]],
        "HighChol": binary_map[raw_data["HighChol"]],
        "CholCheck": binary_map[raw_data["CholCheck"]],
        "BMI": raw_data["BMI"],
        "Smoker": binary_map[raw_data["Smoker"]],
        "Stroke": binary_map[raw_data["Stroke"]],
        "HeartDiseaseorAttack": binary_map[raw_data["HeartDiseaseorAttack"]],
        "Age": raw_data["Age"]
    }

def generate_random_case():
    """Generate a random patient case with risk factors, BMI, and Age"""
    first_part = [random.randint(0,1) for _ in range(3)]  # first 3 risk factors
    bmi = round(random.uniform(13, 34))
    second_part = [random.randint(0,1) for _ in range(3)]  # remaining risk factors
    age = round(random.uniform(18, 80))
    return first_part + [bmi] + second_part + [age]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Diabetes Prediction App")

# Tab layout: User Input vs Random Test
tab1, tab2 = st.tabs(["Manual Input", "Extreme Case Generator"])

# -------- Manual Input --------
with tab1:
    HighBP = st.radio("Do you have High Blood Pressure?", ["Yes", "No"])
    HighChol = st.radio("Do you have High Cholesterol?", ["Yes", "No"])
    CholCheck = st.radio("Have you done a Cholesterol Check?", ["Yes", "No"])
    BMI = st.number_input("Enter your BMI", min_value=10, max_value=50)
    Smoker = st.radio("Are you a Smoker?", ["Yes", "No"])
    Stroke = st.radio("Have you ever had a Stroke?", ["Yes", "No"])
    HeartDiseaseorAttack = st.radio("Any Heart Disease or Attack?", ["Yes", "No"])
    Age = st.number_input("Enter your Age", min_value=18, max_value=120)

    if st.button("Predict Manual Input"):
        raw_data = {
            "HighBP": HighBP,
            "HighChol": HighChol,
            "CholCheck": CholCheck,
            "BMI": BMI,
            "Smoker": Smoker,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "Age": Age
        }

        processed = preprocess_input(raw_data)
        user_df = pd.DataFrame([processed])
        user_scaled = scaler.transform(user_df)
        prediction = loaded_model.predict(user_scaled)[0]
        probability = loaded_model.predict_proba(user_scaled)[0][1]

        st.write("### Result:", "🩸 Diabetic" if prediction==1 else "✅ Not Diabetic")
        st.write("Probability of Diabetes:", round(probability, 2))

# -------- Extreme Case Generator --------
with tab2:
    num_cases = st.number_input("How many random cases?", min_value=1, max_value=20, value=5)
    if st.button("Generate Random Cases"):
        st.write("### Randomly Generated Cases and Predictions:")
        for _ in range(num_cases):
            random_case = generate_random_case()
            # Split into dict to match preprocessing
            raw_data = {
                "HighBP": "Yes" if random_case[0]==1 else "No",
                "HighChol": "Yes" if random_case[1]==1 else "No",
                "CholCheck": "Yes" if random_case[2]==1 else "No",
                "BMI": random_case[3],
                "Smoker": "Yes" if random_case[4]==1 else "No",
                "Stroke": "Yes" if random_case[5]==1 else "No",
                "HeartDiseaseorAttack": "Yes" if random_case[6]==1 else "No",
                "Age": random_case[7]
            }
            processed = preprocess_input(raw_data)
            user_df = pd.DataFrame([processed])
            user_scaled = scaler.transform(user_df)
            prediction = loaded_model.predict(user_scaled)[0]
            probability = loaded_model.predict_proba(user_scaled)[0][1]
            st.write(f"Case: {random_case} → Prediction: {'🩸 Diabetic' if prediction==1 else '✅ Not Diabetic'}, Probability: {round(probability,2)}")
