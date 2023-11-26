# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:13:43 2023

@author: Ajay
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Ajay/Documents/ML-Deployment-1/Traning_model2.sav', 'rb'))

# creating a function for prediction
def diabetic_prediction(input_data):
    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetic Prediction Web App')

    # setting a bright background color
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #ADD8E6; /* Replace with the color of your choice */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # getting the input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetic Prediction Function Value")
    Age = st.text_input("Age of the Person")

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])

    #st.success(diagnosis)
    # styling the result box
    st.markdown(
        f"""
        <div style='border:2px solid #000; border-radius:5px; padding:10px; margin-top:10px;'>
            <p style='font-size:18px; font-weight:bold; color:#000;'>Result:</p>
            <p style='font-size:16px; color:#000;'>{diagnosis}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
