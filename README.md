# 🧑‍💼 Customer Churn Prediction

This project predicts whether a customer will **churn (leave the service)** or **stay** based on their historical data.  
It uses a **Deep Learning model (TensorFlow/Keras)** trained on the [Churn Modelling dataset](https://www.kaggle.com/datasets/shubhendra04/customer-churn-prediction-dataset).

---

## 🚀 Features
- 📊 Preprocessed customer data (encoding categorical variables, scaling numerical features).
- 🤖 Deep Learning model (`model.h5`) trained to predict churn probability.
- 🌐 Interactive **Streamlit App** for easy predictions.
- ✅ Supports categorical inputs like **Geography** and **Gender** using pre-trained encoders.

---

## 📂 Project Structure
churn_prediction/
├── app.py                   # Streamlit web app
├── model.h5                 # Trained deep learning model
├── scaler.pkl               # StandardScaler used during training
├── onehot_encoder_geo.pkl   # OneHotEncoder for 'Geography'
├── label_encoder_gender.pkl # LabelEncoder for 'Gender'
├── Churn_Modelling.csv      # Dataset
├── experiments.ipynb        # Model training experiments
├── prediction.ipynb         # Model prediction testing
├── requirements.txt         # Dependencies
└── README.md                # Project documentation


---

## 🛠️ Installation & Setup

 **Clone the repository**
   ```bash
   git clone https://github.com/kishanyash/churn_prediction.git
   cd churn_prediction
