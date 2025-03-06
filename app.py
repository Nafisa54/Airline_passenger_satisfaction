   # Load the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st

# Load the pre-trained model
import pickle

# Load the dataset
with open (r'C:\Users\NCC\Desktop\Machine_learning_SP\ML_assignment\algo_model2.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# STREAMLIT UI
st.title('Airline Passenger Satisfaction Prediction')
st.write('This app predicts the satisfaction of passengers based on their feedback')
st.write('please wait for the model to load')

# Create a function to predict the satisfaction

def predict_satisfaction(Age, Gender, Type_Travel, Class, Flight_Distance, Inflight_Meal_Quality, Ease_of_Online_Booking, Onboard_Service, Leg_Room_Service, Baggage_Handling, Checkin_Service, Inflight_WiFi_Service, Inflight_Ease_of_Book, Cleanliness, Departure_Delay_In_Minutes, Arrival_Delay_In_Minutes):
      input = np.array (input, dtype=np.float64).reshape(1,-1)
      prediction = loaded_model.predict(input)
      return

# Collect user input features
Age = st.number_input('Age', min_value=0, max_value=100, value
= 0, step=1)

Gender = st.selectbox('Gender', ['Male', 'Female'])

Type_Travel = st.selectbox('Type of Travel', ['Business', 'Personal'])

Class = st.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])

Flight_Distance = st.number_input('Flight Distance', min_value=0, max_value=10000, value=0, step=100)

Inflight_Meal_Quality = st.selectbox('Inflight Meal Quality', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Ease_of_Online_Booking = st.selectbox('Ease of Online Booking', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Onboard_Service = st.selectbox('Onboard Service', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Leg_Room_Service = st.selectbox('Leg Room Service', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Baggage_Handling = st.selectbox('Baggage Handling', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Checkin_Service = st.selectbox('Checkin Service', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Inflight_WiFi_Service = st.selectbox('Inflight WiFi Service', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Inflight_Ease_of_Book = st.selectbox('Inflight Ease of Booking', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Cleanliness = st.selectbox('Cleanliness', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])

Departure_Delay_In_Minutes = st.number_input('Departure Delay in Minutes', min_value=0, max_value=1440, value=0, step=10)

Arrival_Delay_In_Minutes = st.number_input('Arrival Delay in Minutes', min_value=0, max_value=1440, value=0, step=10)

# Make prediction
if st.button('Predict Satisfaction'):
    output = predict_satisfaction
    st.write('Predicted Satisfaction:', output)
    

    # Save the prediction to a file
    with open('C:/Users/NCC/Desktop/Machine_learning_SP/ML_assignment/prediction.txt', 'w') as file:
        file.write(str(output))
        
