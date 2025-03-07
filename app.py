import pickle
import numpy as np
import streamlit as st

# Load the trained model and scaler
model = pickle.load(open('solar_power_system.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_failure(temperature, voltage, current):
    # Calculate the voltage-temp difference (4th feature)
    voltage_temp_diff = voltage - temperature
    
    # Ensure inputs are in the correct shape (a 2D array for model prediction)
    inputs = np.array([[temperature, voltage, current, voltage_temp_diff]])  # Add the calculated feature
    
        
    # Make prediction using the loaded model
    prob = model.predict_proba(inputs)  # Returns probabilities
    
    # Log probabilities for debugging
    print(f"Prediction Probabilities: {prob}")
    
    # If probability of failure is greater than 0.5, predict failure
    if prob[0][1] > 0.5:
        return 1  # Failure
    else:
        return 0  # No Failure


# Streamlit app interface
st.set_page_config(page_title="Solar Power Failure Prediction", page_icon="⚡", layout="wide")

# Header with an image and description
st.title("🌞 Solar Power System Failure Prediction ⚡")
st.write("""
This web application predicts whether your solar power system is likely to fail based on the input values for temperature, voltage, and current.
Use the input fields below to make predictions.
""")

# Sidebar for a clean UI
st.title("Input Features")
st.write("Enter the values for the system to predict failure:")

# User input for the features with better design
temperature = st.number_input("Enter Temperature (°C):", min_value=-50, max_value=50, value=25)
voltage = st.number_input("Enter Voltage (V):", min_value=0, max_value=300, value=230)
current = st.number_input("Enter Current (A):", min_value=0, max_value=100, value=5)

# Add a stylish button with icon
if st.button("Predict Failure 🚀"):
    # Make prediction using the model
    result = predict_failure(temperature, voltage, current)
    
    # Display the result with a beautiful output
    if result == 1:
        st.markdown('<h3 style="color:red; text-align:center;">⚠️ **Failure Detected!** ⚠️</h3>', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color:green; text-align:center;">✅ **No Failure Detected** ✅</h3>', unsafe_allow_html=True)
