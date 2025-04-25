import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, label_encoder

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Reading Behavior Predictor",
        page_icon="📚",
        layout="wide"
    )
    
    # Header
    st.title("📚 Reading Behavior Prediction App")
    st.markdown("### Predict what a reader will look at based on various factors")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a Random Forest model to predict reading behavior based on "
        "demographic information, visual characteristics, and reading environment factors."
    )
    
    # Load model and encoder
    model, label_encoder = load_model()
        
    #Make Prediction
    st.header("Enter Reader Information")
        
    col1, col2 = st.columns(2)
        
    with col1:
        country = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "Germany", "France", "Japan", "China", "India"])
        age = st.number_input("Age", min_value=5, max_value=100, value=30)
        gender = st.selectbox("Gender", ['male', 'female'])
        education = st.selectbox("Education Level", ['undergraduate', 'high school', 'graduate', 'postgraduate'])
        visual_acuity = st.selectbox("Visual Acuity", ['average', 'good', 'poor'])
        
    with col2:
        reading_speed = st.selectbox("Reading Speed", ['normal', 'slow', 'fast'])
        text_density = st.selectbox("Text Density", ['medium', 'low', 'high'])
        font_size = st.selectbox("Font Size", ['medium', 'big', 'small'])
        paper_type = st.selectbox("Paper Type", ['report', 'academic paper', 'book', 'newspaper'])
        initial_focus_time = st.selectbox("Initial Focus Time", ['normal', 'fast', 'slow'])
        
    # Create a button to predict
    if st.button("Predict Reading Behavior"):
        # Create a dataframe from input
        input_data = pd.DataFrame({
            'country': [country],
            'age': [age],
            'gender': [gender],
            'education': [education],
            'visual acuity': [visual_acuity],
            'reading speed': [reading_speed],
            'text density': [text_density],
            'font size': [font_size],
            'paper type': [paper_type],
            'initial focus time': [initial_focus_time]
        })
            
        # Make prediction
        prediction = model.predict(input_data)
           
        # Get the original class name
        #predicted_class = label_encoder.inverse_transform(prediction)[0]
           
        # Display prediction
        st.success(f"### Predicted Reading Behavior: **{prediction[0]}**")


if __name__ == "__main__":
    main()