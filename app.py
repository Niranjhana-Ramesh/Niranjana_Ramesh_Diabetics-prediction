import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Page Config
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🏥", layout="wide")


# Apply background image and overlay
st.markdown("""
    <style>
    /* Set Background Image */
    .stApp {
        background: url("https://4kwallpapers.com/images/wallpapers/windows-11-dark-mode-abstract-background-black-background-3840x2160-8710.jpg") no-repeat center center fixed;
        background-size: cover;
    }

    /* Overlay Effect */
    .main-container {
        position: relative;
        padding: 20px;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.6); /* Dark overlay for contrast */
        color: white !important;
        width: 100%;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(42, 42, 42, 0.9) !important;
        color: white !important;
        border-radius: 10px;
        padding: 20px;
    }

    /* Improve Text Readability */
    h1, h2, h3, h4, h5, h6, p, span {
        color: white !important;
    }

    /* Styled Input Fields */
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid white !important;
        border-radius: 5px !important;
        color: white !important;
    }

    /* Buttons Styling */
    .stButton>button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 16px !important;
    }

    .stButton>button:hover {
        background-color: #0056b3 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Preprocessing function
def preprocess_custom_data(data):
    try:
        preprocessor_impute = joblib.load('models/preprocessor_impute.pkl')
        preprocessor_scale = joblib.load('models/preprocessor_scale.pkl')
    except FileNotFoundError as e:
        st.error(f"Preprocessing file missing: {e}")
        return None

    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    yes_no_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 
                   'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 
                   'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
    for col in yes_no_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

    numerical_cols = ['Age']
    categorical_cols = [col for col in data.columns if col not in numerical_cols]

    X_imputed = preprocessor_impute.transform(data)
    X_imputed_df = pd.DataFrame(X_imputed, columns=numerical_cols + categorical_cols)

    for col in categorical_cols:
        X_imputed_df[col] = X_imputed_df[col].astype(int)

    X_scaled = preprocessor_scale.transform(X_imputed_df)
    return X_scaled

# Model files (excluding MLPClassifier due to version mismatch)
model_files = {
    #'Logistic Regression': 'models/Logistic_Regression_tuned.pkl',
   # 'Decision Tree': 'models/Decision_Tree_tuned.pkl',
    'Random Forest': 'models/Random_Forest_tuned.pkl',
    #'SVM': 'models/SVM_tuned.pkl',
    #'KNN': 'models/KNN_tuned.pkl',
    #git add README.md
# 'XGBoost': 'models/XGBoost_tuned.pkl'
    # 'MLPClassifier': 'models/MLPClassifier_tuned.pkl'  # Commented out due to error
}

# Main content
st.title("🏥 Early Stage Diabetes Risk Prediction")
st.markdown("Enter patient details in the sidebar to predict diabetes risk using advanced machine learning models.")

# Sidebar for input
with st.sidebar:
    st.subheader("Patient Information")
    with st.form(key='diabetes_form'):
        age = st.number_input("Age", min_value=20, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        polyuria = st.selectbox("Polyuria (Excessive Urination)", ["Yes", "No"])
        polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["Yes", "No"])
        weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
        weakness = st.selectbox("Weakness", ["Yes", "No"])
        polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["Yes", "No"])
        genital_thrush = st.selectbox("Genital Thrush", ["Yes", "No"])
        visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
        itching = st.selectbox("Itching", ["Yes", "No"])
        irritability = st.selectbox("Irritability", ["Yes", "No"])
        delayed_healing = st.selectbox("Delayed Healing", ["Yes", "No"])
        partial_paresis = st.selectbox("Partial Paresis (Muscle Weakness)", ["Yes", "No"])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ["Yes", "No"])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])

        submit_button = st.form_submit_button(label="Predict Risk")

# Process input and predict
if submit_button:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Polyuria': [polyuria],
        'Polydipsia': [polydipsia],
        'sudden weight loss': [weight_loss],
        'weakness': [weakness],
        'Polyphagia': [polyphagia],
        'Genital thrush': [genital_thrush],
        'visual blurring': [visual_blurring],
        'Itching': [itching],
        'Irritability': [irritability],
        'delayed healing': [delayed_healing],
        'partial paresis': [partial_paresis],
        'muscle stiffness': [muscle_stiffness],
        'Alopecia': [alopecia],
        'Obesity': [obesity]
    })

    X_scaled = preprocess_custom_data(input_data)
    if X_scaled is None:
        st.error("Preprocessing failed. Cannot proceed with predictions.")
    else:
# Container for results
        st.markdown(
            """
            <div style="
                background-color: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-weight: bold;">
                <h3>Prediction Results</h3>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        for i, (name, filepath) in enumerate(model_files.items()):
            if not os.path.exists(filepath):
                st.error(f"Model file {filepath} not found!")
                continue
            try:
                model = joblib.load(filepath)
                prediction = model.predict(X_scaled)[0]
                result = "Positive (Diabetic)" if prediction == 1 else "Negative (Non-Diabetic)"
                color = "#ffcc00" if prediction == 1 else "#00cc66"

                with col1 if i % 2 == 0 else col2:
                    st.markdown(
                        f'<p style="color: {color}; font-weight: bold;">{name}: {result}</p>',
                        unsafe_allow_html=True
                    )
            except ValueError as e:
                st.error(f"Error loading {name} model: {e}. Possible version mismatch.")

        st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: #ffffff;'>Developed with ❤️ using Streamlit | Data Source: Sylhet Diabetes Hospital</p>
""", unsafe_allow_html=True)