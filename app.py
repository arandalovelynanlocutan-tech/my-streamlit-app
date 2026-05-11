import streamlit as st
import pandas as pd
import pickle

# ================================
# LOAD MODEL + FEATURES
# ================================
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Marks Predictor")

st.write("Fill in student details below:")

# ================================
# USER INPUTS (AUTO GENERATED)
# ================================
inputs = {}

for feature in features:
    inputs[feature] = st.number_input(f"{feature}", value=0.0)

# ================================
# PREDICTION
# ================================
if st.button("Predict Marks"):
    input_df = pd.DataFrame([inputs])

    try:
        prediction = model.predict(input_df)
        st.success(f"📊 Predicted Marks: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
