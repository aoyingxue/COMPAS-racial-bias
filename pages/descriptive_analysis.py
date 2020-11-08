# Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data from cache
@st.cache
def load_data(nrows, dataset_name):
    if dataset_name == "COMPAS Raw Scores":
        path = "data/compas-scores-raw.csv"
    elif dataset_name == "Compass Score after parsing and cleaning":
        path = "data/cox-violent-parsed.csv"
    data = pd.read_csv(path, nrows=nrows)
    # preprocess goes from here
    return data

def descriptive_analysis():
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ("COMPAS Raw Scores", "Compass Score after parsing and cleaning"),
    )
    data = load_data(10000, dataset_name)

    # checkbox
    if st.checkbox("View Chosen Data: %s" % dataset_name):
        st.subheader(dataset_name)
        st.dataframe(data)

    # code box
    # with st.echo():
    #     st.write(test_data.head())

    # slider
    # st.slider()

    with st.beta_container():
        var = st.selectbox("Choose a Variable", data.columns)
        st.write("## Histogram of %s" % var)
        # histogram
        fig = plt.figure()
        plt.hist(data[var])
        plt.xticks(rotation=270)
        st.pyplot(plt)
    # dropdown to select the variable