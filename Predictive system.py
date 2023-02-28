# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/Ajay/Documents/ML-Deployment-1/Traning_model2.sav', 'rb'))

input_data = (1,89,66,23,94,28.1,0.167,21)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
 
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# #Standardize the input data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")