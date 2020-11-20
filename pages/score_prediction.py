# Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Python Scripts
from pages.descriptive_analysis import preprocess_cox
from pages.model import logit_model

## Prediction Page
def run_the_prediction():
    with st.beta_container():
        st.write(
            """
            **Please choose the options most suitable for your current offense.**
            """
        )
        
        ## Variables
        sex = st.sidebar.selectbox("Your sex", ['Female','Male'])
        priors_count=st.sidebar.slider("Count of priors; if you don't have any, please put 0.",min_value=0,max_value=100)
        c_degree=st.sidebar.selectbox("Your current charge",['Felony','Misdemeanor'])
        age=st.sidebar.slider("Your age",min_value=1,max_value=150)
        length_of_stay=st.sidebar.slider("Days of stay in jail",min_value=0,max_value=10000)
        race=st.sidebar.selectbox("Your race",['African-American','Asian','Caucasian','Hispanic','Native American','Others'])
        conviction_more_than_2=st.sidebar.selectbox("Did you have more than 2 times of convictions before?",['No','Yes'])
        
        ## Prepare the dataframe for the prediction
        df=pd.DataFrame(columns=['age','priors_count','c_degree_F','female','length_of_stay',
                                'African-American','Asian','Caucasian','Hispanic','Native American','conviction_more_than_2'])
        ## sex
        if sex=='Female':
            df['female']=[1]
        else: 
            df['female']=[0]
        ## priors_count
        df['priors_count']=priors_count
        ## age
        df['age']=age
        ## length_of_stay
        df['length_of_stay']=length_of_stay
        ## conviction_more_than_2
        if conviction_more_than_2=='No':
            df['conviction_more_than_2']=[0]
        else:
            df['conviction_more_than_2']=[1]
        ## c_degree
        if c_degree=='Felony':
            df['c_degree_F']=[1]
        else: 
            df['c_degree_F']=[0]
        ## race
        if race=='African-American':
            df['African-American']=[1]
        elif race=='Asian':
            df['Asian']=[1]
        elif race=='Caucasian':
            df['Caucasian']=[1]
        elif race=='Hispanic':
            df['Hispanic']=[1]
        elif race=='Native American':
            df['Native American']=[1]
        df.fillna(0,inplace=True)

        ## prediction
        model=logit_model()
        pred=model.predict(df)
        st.write("### Your prediction score is ",pred[0])

    ## future improvements
    st.write(
        """
        -----
        #### Future Improvements
        As the accuracy shown in the last page, the model has its flaws.
        The main reason behind is that the algorithm COMPAS uses has a lot more variables about the person's demographics,
        which is something we don't possess currently.\n
        Based on the publication ***"EVALUATING THE PREDICTIVE VALIDITY OF THE COMPAS RISK AND NEEDS ASSESSMENT SYSTEM"*** by *Northpointe*, 
        the company behind COMPAS Risk Models and markets it to Law Enforcement,
        has multiple COMPAS scales such as:
        - Criminal Involvement
        - History of Violence
        - History of Noncompliance
        - Criminal Associates
        - Substance Abuse
        - Financial Problems and Poverty
        - Occupational and Educational Resources or Human Capital
        - Family Crime
        - High Crime Neighborhood
        - Boredom and Lack of Constructive Leisure Activities (Aimlessness)
        - Residential Instability
        - ...\n
        Due to the time limit, I haven't been able to design a perfect model that can have similar score as the COMPAS Risk Models would give,
        but as a prototype, I have successfully designed a web app that can receive the user's input information and give back to the model to produce a predicted risk.  
        """
    )
