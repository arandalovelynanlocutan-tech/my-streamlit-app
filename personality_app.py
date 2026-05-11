import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ================================
# Load Model
# ================================
with open('personalitymodel.pkl', 'rb') as file:
    model = pickle.load(file)

# ================================
# App Config
# ================================
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Personality Predictor")
st.markdown("Answer the questions below to find out if you're an **Introvert** or **Extrovert**.")
st.divider()

# ================================
# Input Features
# Ranges taken directly from personality.csv
# Time_spent_Alone   : 0 – 11
# Social_event_att.  : 0 – 10
# Going_outside      : 0 – 7
# Friends_circle_size: 0 – 15
# Post_frequency     : 0 – 10
# Stage_fear         : No / Yes
# Drained_after_soc. : No / Yes
# ================================
st.subheader("📋 Tell Us About Yourself")

col1, col2 = st.columns(2)

with col1:
    time_spent_alone = st.slider(
        "🏠 Time Spent Alone (hrs/day)",
        min_value=0, max_value=11, value=4, step=1,
        help="How many hours per day do you spend alone?"
    )

    social_event_attendance = st.slider(
        "🎉 Social Event Attendance (per month)",
        min_value=0, max_value=10, value=4, step=1,
        help="How many social events do you attend per month?"
    )

    going_outside = st.slider(
        "🚶 Going Outside (days/week)",
        min_value=0, max_value=7, value=3, step=1,
        help="How many days a week do you go outside?"
    )

    friends_circle_size = st.slider(
        "👥 Friends Circle Size",
        min_value=0, max_value=15, value=7, step=1,
        help="How many close friends do you have?"
    )

with col2:
    post_frequency = st.slider(
        "📱 Post Frequency (per week)",
        min_value=0, max_value=10, value=5, step=1,
        help="How many times do you post on social media per week?"
    )

    stage_fear = st.selectbox(
        "🎤 Do you have Stage Fear?",
        options=["No", "Yes"],
        help="Do you feel nervous speaking or performing in public?"
    )

    drained_after_socializing = st.selectbox(
        "😓 Drained After Socializing?",
        options=["No", "Yes"],
        help="Do you feel tired or exhausted after spending time with people?"
    )

st.divider()

# ================================
# Encode Categorical
# Training used LabelEncoder — encodes alphabetically
# No = 0, Yes = 1
# ================================
stage_fear_encoded = 1 if stage_fear == "Yes" else 0
drained_encoded = 1 if drained_after_socializing == "Yes" else 0

# ================================
# Build Input — same column order as training
# ================================
input_df = pd.DataFrame([[
    time_spent_alone,
    stage_fear_encoded,
    social_event_attendance,
    going_outside,
    drained_encoded,
    friends_circle_size,
    post_frequency
]], columns=[
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
])

# ================================
# Preprocess — match training pipeline exactly
# ================================
imputer = SimpleImputer(strategy='mean')
input_imputed = imputer.fit_transform(input_df)

scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_imputed)

# ================================
# Predict
# ================================
if st.button("🔍 Predict My Personality", use_container_width=True):
    prediction = model.predict(input_scaled)
    result = prediction[0]  # "Extrovert" or "Introvert"

    st.divider()

    if result == "Extrovert":
        st.success("### 🌟 You are an **Extrovert**!")
        st.markdown("""
        **Extroverts** gain energy from social interactions. You likely enjoy:
        - 🎉 Going to parties and social events
        - 👥 Being around people and making new friends
        - 🎤 Speaking up and being the center of attention
        - 🚀 Taking action and being spontaneous
        """)
    else:
        st.info("### 🌙 You are an **Introvert**!")
        st.markdown("""
        **Introverts** recharge through alone time. You likely enjoy:
        - 📚 Reading, writing, or solo hobbies
        - 🤔 Deep thinking and self-reflection
        - 👫 A small, close circle of meaningful friendships
        - 🏠 Quiet environments over loud, crowded places
        """)

    # ================================
    # Input Summary
    # ================================
    st.divider()
    st.subheader("📊 Your Input Summary")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🏠 Time Alone",          f"{time_spent_alone} hrs/day")
        st.metric("🎉 Social Events",        f"{social_event_attendance}/month")
        st.metric("🚶 Goes Outside",         f"{going_outside} days/week")
        st.metric("👥 Friends Circle",       friends_circle_size)
    with c2:
        st.metric("📱 Post Frequency",       f"{post_frequency}/week")
        st.metric("🎤 Stage Fear",           stage_fear)
        st.metric("😓 Drained After Social", drained_after_socializing)

st.divider()
st.caption("Built with ❤️ using Streamlit | Random Forest Classifier | Dataset: personality.csv")