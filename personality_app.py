import streamlit as st
import pickle
import pandas as pd

# ================================
# Load Model
# ================================
with open('personalitymodel.pkl', 'rb') as f:
    model = pickle.load(f)

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
# All ranges taken directly from personality.csv
#
# EXTROVERT profile (CSV medians):
#   Time_alone=2, Social=6, Outside=5, Friends=9, Post=6
#   Stage_fear=No, Drained=No
#
# INTROVERT profile (CSV medians):
#   Time_alone=7, Social=2, Outside=1, Friends=3, Post=1
#   Stage_fear=Yes, Drained=Yes
# ================================
st.subheader("📋 Tell Us About Yourself")

col1, col2 = st.columns(2)

with col1:
    time_spent_alone = st.slider(
        "🏠 Time Spent Alone (hrs/day)",
        min_value=0, max_value=11, value=4, step=1,
        help="Introvert: usually 5–11 hrs | Extrovert: usually 0–3 hrs"
    )
    social_event_attendance = st.slider(
        "🎉 Social Event Attendance (per month)",
        min_value=0, max_value=10, value=4, step=1,
        help="Introvert: usually 0–3 | Extrovert: usually 5–10"
    )
    going_outside = st.slider(
        "🚶 Going Outside (days/week)",
        min_value=0, max_value=7, value=3, step=1,
        help="Introvert: usually 0–2 days | Extrovert: usually 4–7 days"
    )
    friends_circle_size = st.slider(
        "👥 Friends Circle Size",
        min_value=0, max_value=15, value=7, step=1,
        help="Introvert: usually 0–5 | Extrovert: usually 7–15"
    )

with col2:
    post_frequency = st.slider(
        "📱 Post Frequency (per week)",
        min_value=0, max_value=10, value=4, step=1,
        help="Introvert: usually 0–2 | Extrovert: usually 4–10"
    )
    stage_fear = st.selectbox(
        "🎤 Do you have Stage Fear?",
        options=["No", "Yes"],
        help="92% of Introverts say Yes | 93% of Extroverts say No"
    )
    drained_after_socializing = st.selectbox(
        "😓 Drained After Socializing?",
        options=["No", "Yes"],
        help="92% of Introverts say Yes | 93% of Extroverts say No"
    )

st.divider()

# ================================
# Encode — No=0, Yes=1
# ================================
stage_fear_enc = 1 if stage_fear == "Yes" else 0
drained_enc    = 1 if drained_after_socializing == "Yes" else 0

# ================================
# Build Input — exact column order from training
# ================================
input_df = pd.DataFrame([[
    float(time_spent_alone),
    float(stage_fear_enc),
    float(social_event_attendance),
    float(going_outside),
    float(drained_enc),
    float(friends_circle_size),
    float(post_frequency)
]], columns=[
    'Time_spent_Alone',
    'Stage_fear_enc',
    'Social_event_attendance',
    'Going_outside',
    'Drained_enc',
    'Friends_circle_size',
    'Post_frequency'
])

# ================================
# Predict — no scaling needed (RandomForest)
# ================================
if st.button("🔍 Predict My Personality", use_container_width=True):
    result = model.predict(input_df)[0]

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
