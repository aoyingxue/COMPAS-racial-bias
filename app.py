# Packages
import streamlit as st
import pandas as pd
import numpy as np
import os, urllib
import matplotlib.pyplot as plt

# Python Scripts
import pages.descriptive_analysis as da
import pages.model as md
import pages.score_prediction as spd

# URLError workaround
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# start execution in a main() function
def main():
    # Navigation pages: HOME, Descriptive Analysis, Regression Model, Score Prediction, ABOUT

    # Initialize MARKDOWN Rendering
    home_text = st.markdown(get_file_content_as_string("pages/home.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "",
        [
            "HOME",
            "Descriptive Analysis",
            "Regression Model",
            "Score Prediction",
            "Source Code (ipynb)",
            "ABOUT",
        ],
    )
    if app_mode == "HOME":
        st.sidebar.success("To continue, select the page you'd like to read.")

    elif app_mode == "Descriptive Analysis":
        home_text.empty()  # empty the home page and clear cache
        st.title("Descriptive Analysis")
        da.descriptive_analysis()  # print out the descriptive page

    elif app_mode == "Regression Model":
        home_text.empty()
        st.title("Regression Model")
        md.run_the_model()  # function to run the model

    elif app_mode == "Score Prediction":
        home_text.empty()
        st.title("Score Prediction")
        spd.run_the_prediction()  # function to run the prediction

    elif app_mode == "Source Code (ipynb)":
        home_text.empty()
        st.title("Source Code (ipynb)")
        st.code(get_file_content_as_string("app.py"))  # change to the jupyter notebook

    elif app_mode == "ABOUT":
        home_text.empty()
        about_text = st.markdown(get_file_content_as_string("README.md"))


###### Functions ######

# Read a written markdown content available as a string
@st.cache
def get_file_content_as_string(path):
    with open(path, "r") as file:
        data = file.read()
    return data


# Execute the main function
if __name__ == "__main__":
    main()
