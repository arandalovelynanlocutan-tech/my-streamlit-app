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
# (Ranges based on Student_Marks.csv)
# number_courses: 3 to 8
# time_study: 0.0 to 8.0
# Marks: ~5.6 to ~54.3
# ================================
st.subheader("📋 Enter Student Information")

number_courses = st.slider(
    "📚 Number of Courses",
    min_value=3,
    max_value=8,
    value=5,
    step=1,
    help="Number of courses the student is enrolled in (3–8)."
)

time_study = st.slider(
    "⏱️ Time Spent Studying (hours/day)",
    min_value=0.0,
    max_value=8.0,
    value=4.0,
    step=0.1,
    help="Average hours spent studying per day (0.0–8.0)."
)

st.divider()

# ================================
# Show Input Summary
# ================================
col1, col2 = st.columns(2)
col1.metric("📚 Number of Courses", number_courses)
col2.metric("⏱️ Study Hours/Day", f"{time_study:.1f} hrs")

st.divider()

# ================================
# Prediction
# ================================
if st.button("🔍 Predict Marks", use_container_width=True):
    input_data = np.array([[number_courses, time_study]])
    prediction = model.predict(input_data)
    predicted_mark = round(float(prediction[0]), 2)

    st.success(f"### 📊 Predicted Marks: **{predicted_mark}**")

    # Feedback based on score range from dataset (~5.6 to ~54.3)
    if predicted_mark >= 40:
        st.balloons()
        st.info("🌟 Excellent! This student is expected to perform very well.")
    elif predicted_mark >= 25:
        st.info("👍 Good performance expected. A little more effort goes a long way!")
    elif predicted_mark >= 15:
        st.warning("📖 Average performance. Consider increasing study time.")
    else:
        st.error("⚠️ Low marks predicted. Significant improvement in study habits is recommended.")

st.divider()
st.caption("Built with ❤️ using Streamlit | Linear Regression Model | Dataset: Student_Marks.csv")
