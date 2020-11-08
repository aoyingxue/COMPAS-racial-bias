# Packages
import streamlit as st
import pandas as pd
import numpy as np
import os, urllib
import matplotlib.pyplot as plt

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

# read data from cache
@st.cache
def load_data(nrows,dataset_name):
    if dataset_name=="COMPAS Raw Scores":
        path="data/compas-scores-raw.csv"
    elif dataset_name=="Compass Score after parsing and cleaning":
        path="data/cox-violent-parsed.csv"
    data = pd.read_csv(path, nrows=nrows)
    # preprocess goes from here
    return data

def descriptive_analysis():
    st.title("Descriptive Analysis")
    dataset_name = st.sidebar.selectbox(
        "Select Dataset", ("COMPAS Raw Scores", "Compass Score after parsing and cleaning"))
    data = load_data(10000,dataset_name)

    # checkbox
    if st.checkbox('View Chosen Data: %s' %dataset_name):
        st.subheader(dataset_name)
        st.dataframe(data)


    # code box
    # with st.echo():
    #     st.write(test_data.head())

    # slider
    # st.slider()

    with st.beta_container():
        var = st.selectbox("Choose a Variable", data.columns)
        st.write('## Histogram of %s' % var)
        # histogram
        fig = plt.figure()
        plt.hist(data[var])
        plt.xticks(rotation=270)
        st.pyplot(plt)
    # dropdown to select the variable

## Model Page
def run_the_model():
    st.title("Regression Model")


## Prediction Page
def run_the_prediction():
    st.title("Score Prediction")


# Execute the main function
if __name__ == "__main__":
    main()
