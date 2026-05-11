import streamlit as st
import pickle
import numpy as np

# ================================
# Load Model
# ================================
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# ================================
# App UI
# ================================
st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Marks Predictor")
st.markdown("Predict a student's marks based on their study habits using a **Linear Regression** model.")
st.divider()

# ================================
# Input Features
# ================================
st.subheader("📋 Enter Student Information")

number_courses = st.slider(
    "📚 Number of Courses",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="How many courses is the student currently enrolled in?"
)

time_study = st.number_input(
    "⏱️ Time Spent Studying (hours/day)",
    min_value=0.0,
    max_value=24.0,
    value=4.5,
    step=0.5,
    help="Average number of hours the student studies per day."
)

st.divider()

# ================================
# Prediction
# ================================
if st.button("🔍 Predict Marks", use_container_width=True):
    input_data = np.array([[number_courses, time_study]])
    prediction = model.predict(input_data)
    predicted_mark = round(float(prediction[0]), 2)

    st.success(f"### 📊 Predicted Marks: **{predicted_mark}**")

    # Feedback based on score
    if predicted_mark >= 80:
        st.balloons()
        st.info("🌟 Excellent performance expected! Keep it up!")
    elif predicted_mark >= 60:
        st.info("👍 Good performance expected. A little more effort goes a long way!")
    else:
        st.warning("⚠️ Performance might be low. Consider increasing study time.")

st.divider()
st.caption("Built with ❤️ using Streamlit | Linear Regression Model")
