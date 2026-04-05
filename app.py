import streamlit as st
import pandas as pd
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "ridge_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Put 'ridge_model.pkl' in same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD LOCATIONS (for dropdown)
# =====================================================
# Best way: save locations separately while training
LOCATIONS_PATH = "locations.pkl"

if os.path.exists(LOCATIONS_PATH):
    locations = joblib.load(LOCATIONS_PATH)
else:
    # fallback (if file not present)
    locations = ["Whitefield", "Indira Nagar", "Electronic City", "Marathahalli"]

# =====================================================
# UI DESIGN
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding: 2rem 3rem;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Bangalore House Price Prediction")
st.markdown("### Enter property details")

# =====================================================
# INPUT SECTION
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    sqft = st.number_input("Total Square Feet", min_value=300.0, value=1000.0)
    bath = st.number_input("Bathrooms", min_value=1, value=2)

with col2:
    bhk = st.number_input("BHK", min_value=1, value=2)
    balcony = st.number_input("Balcony", min_value=0, value=1)

with col3:
    location = st.selectbox("Location", sorted(locations))

# =====================================================
# DATAFRAME
# =====================================================
input_data = pd.DataFrame({
    "total_sqft": [sqft],
    "bath": [bath],
    "bhk": [bhk],
    "balcony": [balcony],
    "location": [location]
})

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Predict Price"):
    try:
        prediction = model.predict(input_data)
        st.success("✅ Prediction Successful")
        st.metric("Estimated Price (Lakhs)", round(prediction[0], 2))
    except Exception as e:
        st.error("❌ Prediction Error")
        st.write(e)

# =====================================================
# DEBUG INFO
# =====================================================
with st.expander("⚙️ Debug Info"):
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + Joblib")