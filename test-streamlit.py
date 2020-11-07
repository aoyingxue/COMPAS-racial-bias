import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# big title
st.title("COMPAC Racial Bias")

# description
"""
## Data Description

Racial Bias in inmate COMPAS reoffense risk scores for Florida (ProPublica).

Data, provided by [Kaggle](https://www.kaggle.com/danofer/compass?select=cox-violent-parsed_filt.csv), contains variables used by the COMPAS algorithm in scoring defendants, along with their outcomes within 2 years of the decision, for over 10,000 criminal defendants in Broward County, Florida.

There are mainly 2 different datasets: compas-scores-raw.csv which is the raw scores, cox-violent-parsed.csv which is the Compass Score after some parsing and cleaning.
"""

dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("COMPAS Raw Scores", "Compass Score after parsing and cleaning"))

# read data from cache
@st.cache
def load_data(nrows):
    if dataset_name=="COMPAS Raw Scores":
        path="data/compas-scores-raw.csv"
    elif dataset_name=="Compass Score after parsing and cleaning":
        path="data/cox-violent-parsed.csv"
    data = pd.read_csv(path, nrows=nrows)
    # preprocess goes from here
    return data


data = load_data(10000)

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



