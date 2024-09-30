import streamlit as st
import pandas as pd
import pickle

st.write("""
         # Churn Prediction App
         
         This app predicts bank customer churn
         """)

st.sidebar.header('Input features')

# Coletar novas entradas de dados
def input_features():
        
    credit_score = st.sidebar.slider('Credit Score', 350, 850)
    geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    age = st.sidebar.slider('Age', 18, 100)
    tenure = st.sidebar.slider('Tenure', 0, 10)
    balance = st.sidebar.slider('Balance', 0.0, 250000.0)
    num_products = st.sidebar.slider('Num Of Products', 1, 4)
    has_cr_card = st.sidebar.selectbox('Has Credit Card', ('No', 'Yes'))
    is_active_member = st.sidebar.selectbox('Is Active Member', ('No', 'Yes'))
    estimated_salary = st.sidebar.slider('Estimated Salary', 100.0, 200000.0)

    data = {
        'CreditScore': credit_score,
        'Geography': int(geography.replace('France', '0').replace('Germany', '1').replace('Spain', '2')),
        'Gender': int(gender.replace('Female', '0').replace('Male', '1')),
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_cr_card.replace('No', '0').replace('Yes', '1')),
        'IsActiveMember': int(is_active_member.replace('No', '0').replace('Yes', '1')),
        'EstimatedSalary': estimated_salary
    }
        
    features = pd.DataFrame(data, index=[0])

    return features


df_input_features = input_features()

st.subheader('Input features')

st.write(df_input_features)

# Ler dos dados do modelo construído (arquivo pickle)
classifier_model = pickle.load(open('modelo_churn.pkl', 'rb'))

# Realizar a previsão com base no modelo
prediction = classifier_model.predict(df_input_features)
prediction_proba = classifier_model.predict_proba(df_input_features)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)