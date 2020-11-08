# Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Python Scripts
import pages.descriptive_analysis as da
import pages.model as md

## Prediction Page
def run_the_prediction():
    da.descriptive_analysis()
    return None