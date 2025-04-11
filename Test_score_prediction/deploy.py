import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set page configuration
st.set_page_config(
    page_title="Test Score Predictor",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📊 Student Test Score Prediction")
st.markdown("""
This application predicts student post-test scores based on school and student characteristics.
Enter the information below to get a prediction.
""")

# Load the saved model
def load_model():
    try:
        return joblib.load('Test_score.joblib')
    except:
        st.error("Failed to load model. Please ensure 'Test_score.joblib' is in the current directory.")
        return None

model = load_model()

# Function to make prediction
def predict_score(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Create the input form
st.subheader("Enter Student Information")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    school = st.text_input("School Name", '')
    
    school_setting = st.selectbox(
        "School Setting",
        options=["Urban", "Rural", "Suburban"],
        help="Select the setting of the school"
    )
    
    school_type = st.selectbox(
        "School Type",
        options=["Public", "Non-public"],
        help="Select the type of school"
    )
    
    classroom = st.text_input("Classroom Name", '')

    teaching_method = st.selectbox(
        "Teaching Method",
        options=["Standard", "Experimental"],
        help="Select the primary teaching method used"
    )

with col2:
    n_student = st.slider(
        "Number of Students in Class",
        min_value=10,
        max_value=40,
        value=25,
        help="Select the number of students in the class"
    )
    
    gender = st.selectbox(
        "Student Gender",
        options=["Male", "Female"],
        help="Select the student's gender"
    )
    
    lunch = st.selectbox(
        "Lunch Program",
        options=["Qualifies for reduced/free lunch", "Does not qualify"],
        help="Select the student's lunch program status"
    )
    
    pretest = st.slider(
        "Pre-test Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.5,
        help="Enter the student's pre-test score"
    )

# Create a button for prediction
predict_button = st.button("Predict Post-Test Score", type="primary")

# Prediction section
if predict_button:
    if model is not None:
        # Create input dataframe
        input_data = pd.DataFrame({
            'school': [school],
            'school_setting': [school_setting],
            'school_type': [school_type],
            'classroom': [classroom],
            'teaching_method': [teaching_method],
            'n_student': [n_student],
            'gender': [gender],
            'lunch': [lunch],
            'pretest': [pretest], 
        })
        
        
        try:
            # Make prediction
            with st.spinner("Calculating prediction..."):
                prediction = predict_score(input_data)
            
            # Display prediction with animation
            st.success("Prediction complete!")
            
            # Create columns for prediction display
            pred_col1 = st.columns(1)[0]
            
            with pred_col1:
                # Display the prediction with a metric widget
                st.metric(
                    label="Predicted Post-Test Score",
                    value=f"{prediction:.1f}",
                    delta=f"{prediction - pretest:.1f} from pre-test"
                )        
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("""
            This could be due to:
            1. Input data format not matching what the model expects
            2. Missing features that the model requires
            3. Values outside the range the model was trained on
            """)



# Footer
st.markdown("---")
