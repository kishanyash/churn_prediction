import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="💳", layout="centered")

# --- HEADER ---
st.markdown(
    """
    <h1 style="text-align: center; color: #2E86C1;">💳 Customer Churn Prediction</h1>
    <p style="text-align: center; font-size:18px;">Fill out the customer details below to predict churn probability.</p>
    """,
    unsafe_allow_html=True
)

# --- USER INPUT FORM ---
with st.form("churn_form"):
    st.subheader("📝 Enter Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 30)
        tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
        num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)

    with col2:
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=650)
        balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, step=100.0)
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=100.0)
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
        is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

    # --- CUSTOM BUTTON STYLE ---
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #2E86C1;
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #1B4F72;
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    submitted = st.form_submit_button("🔍 Predict Churn")

# --- PREDICTION LOGIC ---
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Merge
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # --- OUTPUT DISPLAY ---
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**. Take preventive actions!")
    else:
        st.success("✅ The customer is **not likely to churn**. Good relationship maintained.")
