import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Red Zone Project", layout="wide")
st.title("The Red Zone: Clinical Decision Support System")

@st.cache_resource
def load_model():
    try:
        return joblib.load('clinical_xgboost_model.pkl')
    except Exception as e:
        st.error("Model has not been found.")
        st.stop()

model = load_model()

st.sidebar.header("Athlete Telemetry")
st.sidebar.markdown("Adjust the sliders to simulate today's physiological readings.")

# We mainly focus on the features SHAP proved were most important
recovery_score = st.sidebar.slider("Recovery Score (1-100)", 1, 100, 34)
sleep_quality = st.sidebar.slider("Sleep Quality (1-10)", 1, 10, 5)
acute_load = st.sidebar.slider("7-Day Acute Load (EWMA)", 500.0, 5000.0, 3000.0)
chronic_load = st.sidebar.slider("28-Day Chronic Load (EWMA)", 500.0, 5000.0, 2500.0)

acwr = acute_load / chronic_load if chronic_load > 0 else 0

st.sidebar.metric("Calculated ACWR", f"{acwr:.2f}")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### Current Physiological Status")
    st.info(f"**Recovery Score:** {recovery_score}/100\n\n**Sleep Quality:** {sleep_quality}/10\n\n**Acute Load:** {acute_load}")

with col2:
    st.write("### Model Forecast")
    
    if st.button("Run Clinical Prediction", type="primary"):
        expected_features = model.feature_names_in_
        
        input_data = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)
        
        baselines = {
            'heart_rate': 145.0,
            'hydration_level': 65.0,
            'muscle_activity': 200.0,
            'gait_speed': 1.8,
            'cadence': 165.0,
            'step_count': 8000.0,
            'training_load': 1500.0,
            'fatigue_index': 9.0,
            'stress_level': 8.0
        }
        
        for col, val in baselines.items():
            if col in input_data.columns:
                input_data.at[0, col] = val
                
        if 'recovery_score' in input_data.columns: input_data.at[0, 'recovery_score'] = recovery_score
        if 'sleep_quality' in input_data.columns: input_data.at[0, 'sleep_quality'] = sleep_quality
        if 'acute_load_ewma' in input_data.columns: input_data.at[0, 'acute_load_ewma'] = acute_load
        if 'chronic_load_ewma' in input_data.columns: input_data.at[0, 'chronic_load_ewma'] = chronic_load
        if 'acwr_ewma' in input_data.columns: input_data.at[0, 'acwr_ewma'] = acwr
        
        probabilities = model.predict_proba(input_data)[0]
        
        healthy_prob = probabilities[0] * 100
        minor_risk = probabilities[1] * 100
        severe_risk = probabilities[2] * 100
        
        st.write(f"*Raw Prob -> H: {probabilities[0]:.3f} | M: {probabilities[1]:.3f} | S: {probabilities[2]:.3f}*")
        
        st.markdown(f"**Healthy:** {healthy_prob:.1f}%")
        st.progress(float(probabilities[0]))
        
        st.markdown(f"**Minor Injury Risk:** {minor_risk:.1f}%")
        st.progress(float(probabilities[1]))
        
        st.markdown(f"**Severe Injury Risk:** {severe_risk:.1f}%")
        st.progress(float(probabilities[2]))
        
        if severe_risk >= 10.0:
            st.error("**CRITICAL WARNING:** Player is in the Red Zone. Immediate medical intervention required. Reduce training load and prioritize recovery protocols.")
        elif severe_risk >= 2.0:
            st.warning("**CAUTION:** Elevated risk of minor injury. Monitor workload closely today.")
        else:
            st.success("**CLEARED:** Player is highly resilient today. Cleared for maximum training intensity.")