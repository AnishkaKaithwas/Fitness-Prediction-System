
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NEXFIT AI",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

try:
    df = pd.read_csv("gym_members_exercise_tracking.csv")
except:
    st.error("CSV file not found")
    st.stop()

# =====================================================
# CLEAN COLUMNS
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# LOAD MODEL
# =====================================================

model = None

try:
    model = joblib.load("linear_model.pkl")
except:
    try:
        with open("linear_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        st.warning("Model could not load. Prediction section may not work.")

# =====================================================
# COLUMN NAMES
# =====================================================

AGE_COL = "Age"
GENDER_COL = "Gender"
WEIGHT_COL = "Weight (kg)"
HEIGHT_COL = "Height (m)"
SESSION_COL = "Session_Duration (hours)"
CALORIES_COL = "Calories_Burned"
WATER_COL = "Water_Intake (liters)"
WORKOUT_FREQ_COL = "Workout_Frequency (days/week)"
EXP_COL = "Experience_Level"
BMI_COL = "BMI"
WORKOUT_TYPE_COL = "Workout_Type"

# =====================================================
# CSS DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#020617,#0f172a,#111827);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#111827);
    border-right: 2px solid cyan;
}

.big-title {
    font-size: 70px;
    font-weight: 900;
    text-align: center;
    color: #38bdf8;
    text-shadow: 0px 0px 25px cyan;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {opacity:0.7;}
    to {opacity:1;}
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
    text-align:center;
}

.card h1 {
    color: cyan;
    font-size: 40px;
}

.card h2 {
    color: white;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#06b6d4,#2563eb);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 14px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px cyan;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1048/1048941.png",
    width=170
)

page = st.sidebar.radio(
    "⚡ NEXFIT AI CONTROL",
    [
        "🏠 Command Center",
        "🤖 AI Predictor",
        "🌌 3D Analytics",
        "🔥 Heat Intelligence",
        "📊 Statistics Lab",
        "🧾 Data Matrix"
    ]
)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="big-title">🚀 NEXFIT AI SYSTEM</div>',
    unsafe_allow_html=True
)

st.write("")

# =====================================================
# COMMAND CENTER
# =====================================================

if page == "🏠 Command Center":

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'''
        <div class="card">
        <h2>👥 USERS</h2>
        <h1>{len(df)}</h1>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="card">
        <h2>🔥 CALORIES</h2>
        <h1>{round(df[CALORIES_COL].mean(),2)}</h1>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="card">
        <h2>🏋️ SESSION</h2>
        <h1>{round(df[SESSION_COL].mean(),2)}</h1>
        </div>
        ''', unsafe_allow_html=True)

    with c4:
        st.markdown(f'''
        <div class="card">
        <h2>⚡ BMI</h2>
        <h1>{round(df[BMI_COL].mean(),2)}</h1>
        </div>
        ''', unsafe_allow_html=True)

    st.write("")

    # HISTOGRAM
    fig1 = px.histogram(
        df,
        x=CALORIES_COL,
        nbins=40,
        color_discrete_sequence=["cyan"],
        template="plotly_dark",
        title="🔥 Calories Burn Distribution"
    )

    fig1.update_layout(height=500)

    st.plotly_chart(fig1, use_container_width=True)

    # BOX PLOT
    fig2 = px.box(
        df,
        x=GENDER_COL,
        y=CALORIES_COL,
        color=GENDER_COL,
        template="plotly_dark",
        title="📦 Calories Box Plot"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =====================================================
# AI PREDICTOR
# =====================================================

elif page == "🤖 AI Predictor":

    st.subheader("🤖 Smart AI Prediction")

    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age", 10, 80, 25)
        weight = st.slider("Weight (kg)", 30.0, 150.0, 70.0)
        height = st.slider("Height (m)", 1.0, 2.5, 1.75)
        max_bpm = st.slider("Max BPM", 100, 220, 180)
        avg_bpm = st.slider("Average BPM", 60, 200, 140)

    with c2:
        resting_bpm = st.slider("Resting BPM", 40, 100, 65)
        session = st.slider("Session Duration", 0.5, 5.0, 1.5)
        fat = st.slider("Fat Percentage", 5.0, 40.0, 20.0)
        water = st.slider("Water Intake", 1.0, 10.0, 3.0)
        workout_freq = st.slider("Workout Frequency", 1, 7, 3)

    experience = st.slider("Experience Level", 1, 5, 2)

    bmi = st.slider("BMI", 10.0, 50.0, 22.0)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    workout_type = st.selectbox(
        "Workout Type",
        ["Yoga", "HIIT", "Cardio", "Strength"]
    )

    # =========================================
    # ENCODING
    # =========================================
    gender_male = 1 if gender == "Male" else 0

    workout_cardio = 1 if workout_type == "Cardio" else 0
    workout_hiit = 1 if workout_type == "HIIT" else 0
    workout_strength = 1 if workout_type == "Strength" else 0

    # =========================================
    # FINAL 17 FEATURES
    # =========================================

    input_data = np.array([[
    age,
    gender_male,
    weight,
    height,
    max_bpm,
    avg_bpm,
    resting_bpm,
    session,
    fat,
    water,
    workout_freq,
    experience,
    bmi,
    workout_cardio,
    workout_hiit,
    workout_strength,
    0
]])

    if st.button("🚀 RUN AI PREDICTION"):

        try:

            prediction = model.predict(input_data)[0]

            st.success(
                f"🔥 Estimated Calories Burned: {prediction:.2f}"
            )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Calories Burned"},
                gauge={
                    'axis': {'range': [0, 2000]},
                    'bar': {'color': "cyan"}
                }
            ))

            gauge.update_layout(
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
# =====================================================
# 3D ANALYTICS
# =====================================================

elif page == "🌌 3D Analytics":

    st.subheader("🌌 Advanced 3D Analytics")

    fig3d = px.scatter_3d(
        df,
        x=AGE_COL,
        y=SESSION_COL,
        z=CALORIES_COL,
        color=WORKOUT_TYPE_COL,
        size=CALORIES_COL,
        template="plotly_dark",
        title="🚀 3D Fitness Galaxy"
    )

    fig3d.update_layout(height=800)

    st.plotly_chart(fig3d, use_container_width=True)

    # 3D SURFACE
    z_data = np.outer(
        df[SESSION_COL][:50],
        df[CALORIES_COL][:50]
    )

    surface = go.Figure(data=[go.Surface(z=z_data)])

    surface.update_layout(
        title="🌊 3D Surface Visualization",
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(surface, use_container_width=True)

# =====================================================
# HEATMAP
# =====================================================

elif page == "🔥 Heat Intelligence":

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Turbo",
        template="plotly_dark"
    )

    heat.update_layout(height=800)

    st.plotly_chart(heat, use_container_width=True)

# =====================================================
# STATISTICS
# =====================================================

elif page == "📊 Statistics Lab":

    st.subheader("📊 Statistical Dashboard")

    scatter = px.scatter(
        df,
        x=SESSION_COL,
        y=CALORIES_COL,
        color=WORKOUT_TYPE_COL,
        size=BMI_COL,
        template="plotly_dark",
        title="📈 Smart Scatter Plot"
    )

    scatter.update_layout(height=600)

    st.plotly_chart(scatter, use_container_width=True)

    violin = px.violin(
        df,
        y=CALORIES_COL,
        x=GENDER_COL,
        color=GENDER_COL,
        box=True,
        template="plotly_dark",
        title="🎻 Violin Distribution"
    )

    violin.update_layout(height=600)

    st.plotly_chart(violin, use_container_width=True)

# =====================================================
# DATA MATRIX
# =====================================================

elif page == "🧾 Data Matrix":

    st.subheader("🧾 Dataset Viewer")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 DOWNLOAD DATASET",
        df.to_csv(index=False),
        "fitness_dataset.csv",
        "text/csv"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h3 style='color:cyan;'>
    ⚡ NEXFIT AI • FUTURE FITNESS DASHBOARD
    </h3>
    </center>
    """,
    unsafe_allow_html=True
)
