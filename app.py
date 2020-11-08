# Packages
import streamlit as st
import pandas as pd
import numpy as np
import os, urllib

# URLError workaround
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# start execution in a main() function
def main():
    # Navigation pages: HOME, Descriptive Analysis, Regression Model, Score Prediction, ABOUT
    
    # Initialize MARKDOWN Rendering
    home_text=st.markdown(get_file_content_as_string("pages/home.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("",
        ["HOME", "Descriptive Analysis", "Regression Model", "Score Prediction", "ABOUT"])
    if app_mode == "HOME":
        st.sidebar.success("To continue, select the page you'd like to read.")
    elif app_mode == "Descriptive Analysis":
        home_text.empty() # empty the home page 
        descriptive_analysis() # print out the descriptive page
        # st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Regression Model":
        home_text.empty() # empty the home page 
        # function to run the model
        run_the_model()
    elif app_mode == "Score Prediction":
        home_text.empty() # empty the home page 
        # function to run the prediction
        run_the_prediction()
    elif app_mode == "ABOUT":
        home_text.empty() # empty the home page 
        about_text=st.markdown(get_file_content_as_string("pages/about.md"))


###### Functions ######

# Read a written markdown content available as a string
@st.cache()
def get_file_content_as_string(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def descriptive_analysis():
    st.title("Descriptive Analysis")

def run_the_model():
    st.title("Regression Model")

def run_the_prediction():
    st.title("Score Prediction")


# Execute
if __name__ == "__main__":
    main()
