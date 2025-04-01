import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Airbnb Price Optimizer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header-text {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #FF5A5F !important;
    }
    .st-bq {
        border-left-color: #FF5A5F !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.preprocessor = None

# Model loading function
def load_model():
    try:
        # Set TF environment variables before importing
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        from tensorflow.keras.models import load_model
        from tensorflow.keras.backend import clear_session
        
        clear_session()  # Clear previous sessions
        
        # Load model with error handling
        model_path = 'models/airbnb_price_model.h5'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None, None
            
        model = load_model(model_path, compile=False)
        
        # Load preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        if not os.path.exists(preprocessor_path):
            st.error(f"Preprocessor file not found at {preprocessor_path}")
            return None, None
            
        preprocessor = joblib.load(preprocessor_path)
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model only once when app starts
if not st.session_state.model_loaded:
    with st.spinner("Loading AI model..."):
        st.session_state.model, st.session_state.preprocessor = load_model()
        st.session_state.model_loaded = True

# App header
st.markdown('<p class="header-text">🏠 Airbnb Price Optimizer</p>', unsafe_allow_html=True)
st.write("AI-powered pricing recommendations for your listings")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    if st.session_state.model:
        st.success("Model loaded successfully!")
    else:
        st.warning("Running in demo mode - no model loaded")
    
    st.subheader("Hyperparameters")
    learning_rate = st.selectbox("Learning rate", [0.001, 0.01, 0.1])
    batch_size = st.selectbox("Batch size", [32, 64, 128])
    
    if st.button("Retrain Model"):
        with st.spinner("Retraining..."):
            # Placeholder for retraining logic
            time.sleep(2)
            st.success("Model retrained!")

# Main content
tab1, tab2 = st.tabs(["Predict", "Analyze"])

with tab1:
    st.header("Price Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Latitude", value=48.8566)
            longitude = st.number_input("Longitude", value=2.3522)
            accommodates = st.slider("Guests", 1, 10, 2)
            
        with col2:
            bedrooms = st.slider("Bedrooms", 0, 5, 1)
            bathrooms = st.slider("Bathrooms", 0.5, 5.0, 1.0)
            min_nights = st.slider("Min nights", 1, 30, 2)
        
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        if st.session_state.model and st.session_state.preprocessor:
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'accommodates': [accommodates],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'minimum_nights': [min_nights]
                })
                
                # Preprocess and predict
                processed = st.session_state.preprocessor.transform(input_data)
                prediction = st.session_state.model.predict(processed)[0][0]
                
                st.success(f"Recommended price: ${prediction:.2f}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            # Demo mode
            st.warning("Using demo prediction - no model loaded")
            st.info("Recommended price: $120.00 (demo)")

with tab2:
    st.header("Market Analysis")
    st.write("Historical pricing data and trends")
    # Add your analysis visualizations here

# Footer
st.markdown("---")
st.markdown("""
**Note**: This app uses an AI model to predict optimal pricing.
Actual results may vary based on market conditions.
""")
