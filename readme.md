# Airbnb Price Prediction Dashboard

This Streamlit app provides AI-powered price recommendations for Airbnb listings with hyperparameter tuning capabilities.

## Deployment Requirements

1. **Python Version**: 3.10 (specified in runtime.txt)
2. **Required Files**:
   - `models/airbnb_price_model.h5` (your trained model)
   - `models/preprocessor.pkl` (your preprocessing pipeline)

## Streamlit Cloud Deployment

1. **First-Time Setup**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main

## Deployment Notes

1. **Python Version**: This app requires Python 3.10 (specified in runtime.txt)
2. **TensorFlow**: Using tensorflow-cpu 2.16.1 for compatibility
3. **Model Files**: Ensure your model files are in the `models/` directory
4. **Deployment Steps**:
   - Push to GitHub
   - On Streamlit Cloud:
     - Select Python 3.10 as runtime
     - Set main file to `app.py`
     - Deploy!

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
