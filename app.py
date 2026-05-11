# ================================
# STREAMLIT ML APP
# ================================
import streamlit as st
import pandas as pd
import pickle

# ================================
# LOAD MODEL
# ================================
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("❌ model.pkl not found. Make sure it's in the same folder.")
    st.stop()

# ================================
# APP TITLE
# ================================
st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")

st.title("🎓 Student Marks Predictor")
st.write("Enter student details to predict marks")

# ================================
# INPUT FIELDS
# ================================
study_hours = st.number_input("📚 Study Hours", min_value=0.0, step=0.5)
attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0)
assignments = st.number_input("📝 Assignments Completed", min_value=0, step=1)

# ================================
# PREDICTION
# ================================
if st.button("🔮 Predict Marks"):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance],
        "Assignments": [assignments]
    })

    try:
        prediction = model.predict(input_data)
        st.success(f"📊 Predicted Marks: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")