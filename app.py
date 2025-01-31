import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
import numpy as np


# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

# Define the feature names
binary_columns = ['Partner', 'Dependents', 'PaperlessBilling']
multi_category_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                          'Contract', 'PaymentMethod']
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Function to preprocess input data
def preprocess_input(data, label_encoders, scaler, onehot_encoder):
    # Label encode binary columns
    for column in binary_columns:
        if column in label_encoders:
            data[column] = label_encoders[column].transform(data[column])

    # One-hot encode categorical columns
    data_encoded = onehot_encoder.transform(data[multi_category_columns])
    data_encoded = pd.DataFrame(data_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(multi_category_columns))

    # Concatenate the encoded data with the original DataFrame
    data = pd.concat([data.drop(multi_category_columns, axis=1), data_encoded], axis=1)

    # Standardize the entire DataFrame
    data = scaler.transform(data)

    return data

# Input form in the center of the page
st.header('Enter Customer Information')

# Create four columns for input fields
col1, col2, col3, col4 = st.columns(4)

with col1:
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'No phone service', 'Yes'])
    online_security = st.selectbox('Online Security', ['No', 'No internet service', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No', 'No internet service', 'Yes'])


with col2:
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_backup = st.selectbox('Online Backup', ['No', 'No internet service', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No', 'No internet service', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'No internet service', 'Yes'])


with col3:
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'No internet service', 'Yes'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    
with col4:
    monthly_charges = st.slider('Monthly Charges', min_value=0.0, max_value=200.0, step=0.01)
    tenure = st.slider('Tenure', min_value=0, max_value=72, step=1)
    total_charges = st.slider('Total Charges', min_value=0.0, max_value=10000.0, step=0.01)

# Button to submit the input data
if st.button('Enter Data'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Preprocess the input data
    input_data = preprocess_input(input_data, label_encoders, scaler, onehot_encoder)

    # Make a prediction
    prediction = model.predict(input_data)
    churn_probability = prediction[0][0]

    # Display the prediction
    st.write(f'Churn Probability: {churn_probability * 100:.2f}%')
    if churn_probability >= 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

