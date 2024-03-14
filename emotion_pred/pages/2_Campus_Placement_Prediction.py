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
model_file_path = os.path.join(model_folder, 'knn_campus_placement_pred.pkl')

# Open the file
with open(model_file_path, 'rb') as model_file:

    loaded_model = pickle.load(model_file)
st.image(r'https://innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png')
st.title('Welcome to Campus Placement Status Prediction App')



def main():

    st.subheader("Enter the details")

    # Create a form using st.form
    with st.form("user_input_form"):
        # Add form components
        ssc = st.text_input("Enter your SSC Percentage: ")
        inter = st.text_input('Enter your 12th Percentage: ')
        grad = st.text_input('Enter your Graduation Precentage: ')
        mba = st.text_input('Enter your MBA Percentage: ')
        exp = st.radio('Do you have Work Experiance: ',options=['Yes','No'])
        spec = st.radio('Select your Specialization:',options=['Mkt&Fin', 'Mkt&HR'])
        # Add a submit button to the form
        submit_button = st.form_submit_button(label="Predict")

    # Process the form submission when the submit button is clicked
    if submit_button:
        # Display the user input after submission
        n = [ssc, inter, grad, mba,exp, spec]
        numeric_features = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p','workex', 'specialisation']
        new_data = pd.DataFrame([n], columns=numeric_features)

        # Use the loaded model to make predictions
        # Note: Ensure that the input format is compatible with the model's expectations
        # For example, if the model expects numerical features, convert the text input accordingly
        predictions = loaded_model.predict(new_data)

        # Display the prediction result
        if predictions[0]==1:
            st.header('Prediction: Placed')
        else:
            st.header('Prediction: Not Placed')       
    #[7465.0,1921.214415,130247.0,0.057314,97.492221,0.594490]
if __name__ == "__main__":
    main()
