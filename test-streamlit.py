import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# big title
st.title("COMPAC Racial Bias")

# description
"""
    
"""

# read data from cache
@st.cache
def load_data(nrows,path):
    data=pd.read_csv(path,nrows=nrows)
    # preprocess goes from here
    return data

compas_scores_raw=load_data(10000,"data/compas-scores-raw.csv")

# checkbox
if st.checkbox('View Data'):
    st.subheader('Raw Data of Compas Score')
    st.write(compas_scores_raw)


# code box
# with st.echo():
#     st.write(test_data.head())

#slider
# st.slider()

# dropdown to select the variable
var=st.sidebar.selectbox("Choose a Variable",compas_scores_raw.columns)


# histogram
'## Histogram of %s' %var
fig=plt.figure()
plt.hist(compas_scores_raw[var])
plt.xticks(rotation=270)
st.pyplot(plt)

