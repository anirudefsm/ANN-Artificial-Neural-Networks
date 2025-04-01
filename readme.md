# Airbnb Price Prediction Dashboard

This Streamlit app provides AI-powered price recommendations for Airbnb listings with hyperparameter tuning capabilities.

## Features

- Price prediction based on property features
- Neural network hyperparameter tuning
- Market insights and analytics
- Beautiful, responsive UI

## Deployment

1. **Prepare your model files**:
   - Place your trained Keras model as `models/model.h5`
   - Place your preprocessing pipeline as `models/preprocessor.pkl`

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt



## Key Features

1. **Model Integration**:
   - Properly loads your saved Keras model and preprocessor
   - Handles prediction with proper feature preprocessing
   - Graceful fallback if model files are missing

2. **Beautiful UI**:
   - Custom CSS styling
   - Responsive layout
   - Professional metrics display
   - Interactive visualizations

3. **Hyperparameter Tuning**:
   - Dynamic sliders for neural network architecture
   - Training parameter controls
   - Simulated retraining functionality

4. **Deployment Ready**:
   - Clear requirements file
   - Proper file structure
   - Complete documentation

To use your actual model, simply place your `model.h5` and `preprocessor.pkl` files in the `models/` directory. The app will automatically detect and use them.
