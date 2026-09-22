#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle


# In[3]:


log_model=pickle.load(open('log.pkl','rb'))


# In[5]:


st.title('Model Deployment using Logistic Regression')


# In[27]:


def user_input_parameter():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20)
    glucose = st.sidebar.number_input("Glucose", 0, 200)
    blood_pressure = st.sidebar.number_input("BloodPressure", 0, 140)
    skin_thickness = st.sidebar.number_input("SkinThickness", 0, 100)
    insulin = st.sidebar.number_input("Insulin", 0, 900)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.sidebar.number_input("Age", 0, 100)
    data = {'Pregnancies': pregnancies,'Glucose': glucose,'BloodPressure': blood_pressure,'SkinThickness': skin_thickness,'Insulin': insulin,'BMI': bmi,'DiabetesPedigreeFunction': dpf,'Age': age}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_parameter()
prediction =log_model.predict(input_df)
prediction_prob = log_model.predict_proba(input_df)
button=st.button('predict')
if button==True:
   st.subheader("Prediction Probabilities")
   st.write(f"Probability of Not Diabetic: {prediction_prob[0]*100:.2f}%")
   st.write(f"Probability of Diabetic    : {prediction_prob[1]*100:.2f}%")
   st.subheader("Final Prediction")
   if prediction == 1:
      st.success("✅ The patient is predicted to be **Diabetic**.")
   else:
      st.success("✅ The patient is predicted to be **Not Diabetic**.")


# In[ ]:




