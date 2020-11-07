# import packages
import streamlit as st
import pandas as pd
import numpy as np
import os, urllib

# start execution in a main() function
def main():
    # Navigation pages: HOME, Descriptive Analysis, Regression Model, Score Prediction, ABOUT
    
    # Render MARKDOWN 
    readme_text = st.markdown(get_file_content_as_string("README.md"))


    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

if __name__ == "__main__":
    main()


# Functions
# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/aoyingxue/COMPAS-racial-bias/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")