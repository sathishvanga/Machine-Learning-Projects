import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Get the absolute path to the current script
script_path = os.path.abspath(__file__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(script_path))

# Specify the path to the model folder relative to the script's directory
model_folder = "model"

# Construct the full path to the model file
model_file_path = os.path.join(model_folder, 'gaussian_brain.pkl')

# Open the file
with open(model_file_path, 'rb') as model_file:

    loaded_model = pickle.load(model_file)

st.title('Welcome to Brain Tumor Prediction App')

# Take user input
# new_data = st.text_input("Enter your text:")

def main():
    global new_data
    st.subheader("Enter the values")

    # Create a form using st.form
    with st.form("user_input_form"):
        # Add form components
        area = st.number_input("Enter area of tumor :")
        perimeter = st.number_input("Enter perimeter :")
        convex_area = st.number_input("Enter convex area :")
        solidity = st.number_input("Enter solidity :")
        diameter = st.number_input("Enter diameter :")
        eccentricity = st.number_input("Enter Eccentricity :")
        # Add a submit button to the form
        submit_button = st.form_submit_button(label="Predict")

    # Process the form submission when the submit button is clicked
    if submit_button:
        # Display the user input after submission
        n = np.array([area, perimeter, convex_area, solidity, diameter, eccentricity])
        numeric_features = ['area', 'perimeter', 'convex_area', 'solidity', 'equivalent_diameter', 'eccentricity']
        new_data = pd.DataFrame([n], columns=numeric_features)

        # Use the loaded model to make predictions
        # Note: Ensure that the input format is compatible with the model's expectations
        # For example, if the model expects numerical features, convert the text input accordingly
        predictions = loaded_model.predict(new_data)

        # Display the prediction result
        st.header(f"Prediction : {predictions[0]}")
       
    #[7465.0,1921.214415,130247.0,0.057314,97.492221,0.594490]
if __name__ == "__main__":
    main()
