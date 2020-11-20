# Packages
import streamlit as st

# Models
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date

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

@st.cache
def preprocess_for_modeling(cox_df):
    df_2yrs=pd.read_csv("data/propublica_raw_data/compas-scores-two-years.csv",parse_dates=['c_jail_out','c_jail_in'])
    df_2yrs=df_2yrs[['age','c_charge_degree','race','age_cat', 'score_text', 'sex', 'priors_count', 
                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
    df_2yrs=df_2yrs.loc[(df_2yrs.days_b_screening_arrest<=30) & 
                        (df_2yrs.days_b_screening_arrest>=-30) & 
                        (df_2yrs.is_recid!=-1) & (df_2yrs.c_charge_degree!='O') & 
                        (df_2yrs.score_text!='N/A'),]
    df_2yrs['length_of_stay']=(df_2yrs.c_jail_out-df_2yrs.c_jail_in).apply(lambda x: x.days)

    df_2yrs=pd.concat([df_2yrs,pd.get_dummies(df_2yrs.c_charge_degree,prefix='c_degree',prefix_sep='_')],axis=1)
    df_2yrs['female']=df_2yrs.sex.apply(lambda x: 1 if x=='Female' else 0)
    df_2yrs=pd.concat([df_2yrs,pd.get_dummies(df_2yrs.race)],axis=1)
    df_2yrs['score_level']=df_2yrs.score_text.apply(lambda x: 0 if x=='Low' else (1 if x=='Medium' else 2))
    return df_2yrs

## Model Page
def run_the_model():
    
    ## read data
    cox_df = preprocess_cox()

    ## overview
    st.write(
        """
        ## Overview
        We use the raw score dataset to run several models including linear regression, 
        multinomial logistic regression, support vector machine and random forest, 
        with "decile scores" as the dependent variable, 
        and age, race, charge degree,  recidivism charge degree, and violent charge degree as independent variables,
        to try to analyze how these factors affect the score. 
        We could identify important factors among these variables in the analysis down below, 
        however, we have not yet developed a model which could perfectly predict the score based on given information. 
        Prediction and the direction of future improvements will be presented in the next page.
        """
    )

    ## preprocessed data
    st.write(
        """
        ## Preprocess for modeling 
        *Reference: Propublica's preprocessed dataset*
        - Use 2-year COMPAS data
        - Remove the row if the charge date of a defendants COMPAS scored crime was not within 30 days from when the person was arrested, which may not stand as a right offense.
        - If recidivist flag -- is_recid -- is -1, there is not a COMPAS Case at all, which needs to be removed.
        - Filtered to include only those rows representing people who had either recidivated in two years, 
        or had at least two years outside of a correctional facility.
        - Filtered to exclude ordinary traffic offenses "O" in the "charge_degree" will not result in Jail.
        - Create a variable "length of stay" because higher COMPAS scores are slightly correlated with a longer length of stay.
        """
    )
    df_2yrs=preprocess_for_modeling(cox_df)

    df2=df_2yrs[['score_level','age','priors_count','c_degree_F','female','length_of_stay',\
                'African-American','Asian','Caucasian','Hispanic','Native American']]
    df3=df_2yrs[['two_year_recid','age','priors_count','c_degree_F','female','length_of_stay',\
                'African-American','Asian','Caucasian','Hispanic','Native American']]

    ## Prior Felony Convictions
    st.write(
        """
        ### Prior Felony Convictions
        >People in Florida who have previously been convicted of two or more felonies and are convicted of yet another felony 
        may be sentenced to a lengthy prison term under one of Florida’s recidivist sentencing schemes. 
        Such laws are often referred to as “Three Strikes and You’re Out” laws. (Fla. Stat. § 775.084 (2019).)
        """
    )
    df2=df2.copy()
    df3=df3.copy()
    df2['conviction_more_than_2']=df2.priors_count.apply(lambda x: 1 if x>=2 else 0)
    df3['conviction_more_than_2']=df3.priors_count.apply(lambda x: 1 if x>=2 else 0)

    st.write("### Data for the modeling")
    st.dataframe(df3.head())

    ## training and testing dataset
    X2=df2.loc[:,~df2.columns.isin(['score_level'])]
    y2=df2.score_level
    
    X3=df3.loc[:,~df3.columns.isin(['two_year_recid'])]
    y3=df3.two_year_recid
    X3_train, X3_test, y3_train, y3_test= train_test_split(X3, y3, test_size=0.2, random_state=1)

    ## Multi-collinearity
    st.write(
        """
        ## Check for multicollinearity
        In some trials, I ran into occasions when variables have serious multicollinearity issues. 
        I referred to VIF scores to identify the issue. 
        If VIF score is too high for a pair of variables or more, it means there will be multicollinearity problem,
        which means that we need to drop one of the variables. 
        """
    )
    
    st.write(vif_scores(X3))
    st.write(
        """
        In this case, there's no obvious multicollinearity issue among different variables.
        """
    )
    
    
    ## Model    
    st.write("## Logistic Regression Model")
    st.write(
            """
            ### Limitations
            Due to the limited time and effort, I tried some of the simple models, 
            such as Linear Regression *(using scores as continuous outcome variable)*, 
            Logistic Regression *(using two_year_recid as binary outcome variable)*, 
            Multinomial Logistic Regression and Support Vector Machine *(using either 10-level scores or score_text as multi-level variable)*.\n
            However, due to the limited number of useful variables, the accuracy scores are far less to be considered as a good model. \n
            Therefore, only the logistic regression with two_year_recid as the binary outcome variable is shown at the page.
            Other models can be read in the source code if you're interested.
            """
        )
    with st.beta_container():
        ## code box
        st.write("""### Accuracy Score""")
        with st.echo():
            model=LogisticRegressionCV(max_iter=10000,cv=10,random_state=42,penalty='l2',solver='lbfgs').fit(X3_train,y3_train.values.ravel())
            pred3=model.predict(X3_test)
            accuracy=metrics.accuracy_score(y3_test,pred3)
        st.write("The accuracy score of the prediction is ",round(accuracy,3),".")

        st.write("### ROC Curve")
        # generate a no skill prediction (majority class)
        ns_probs = [1 for _ in range(len(y3_test))]

        # predict probabilities
        lr_probs = model.predict_proba(X3_test)

        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        # calculate scores
        ns_auc = roc_auc_score(y3_test, ns_probs)
        lr_auc = roc_auc_score(y3_test, lr_probs)

        # summarize scores
        st.write('No Skill: ROC AUC=%.3f' % (ns_auc))
        st.write('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y3_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y3_test, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        st.pyplot(plt)
        st.write(
            """
            From the ROC curve, the area under the curve is 0.723. 
            In other words, the logistic regression model is correctly classifying 
            whether the given person would recidivate within 2 years 72.3% of the time.
            """
        )


    # # slider
    # # st.slider()
    st.write("## Interpretation of the results")
    st.write(
        """
        To interpret the results from the logistic regression, I instead used statsmodels to run a similar regression 
        (I used 3 score levels - *low, medium, high* - as the outcome variable instead of binary indicator of 2 year recidivism),
        because statsmodels provide a detailed summary of the regression model.
        """
    )
    with st.echo():
        logit_model=sm.MNLogit(y2,sm.add_constant(X2))
        result=logit_model.fit()
        stats=result.summary()
    st.text(stats)
    st.write(
        """
        From the regression result, 
        "Asian" is statistically insignificant in predicting both level 1 and level 2, 
        which means that no matter the defendant is Asian or not, 
        the probability of being categorized into either low or high score is the same for the same person.\n
        Similarly, the same goes to Hispanic. For the hispanic defendants, 
        the probability of being categorized into either level of score is the same.\n
        At the meantime, being an African-American or a Caucasian is statistically significant
        in predicting one's risk score, which is the evidence of the disparity shown in the exploratory analysis.
        """
    )


def vif_scores(df):
        VIF_Scores = pd.DataFrame()
        VIF_Scores["Independent Features"] = df.columns
        VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
        return VIF_Scores

def logit_model():
    cox_df = preprocess_cox()
    df_2yrs=preprocess_for_modeling(cox_df)
    df3=df_2yrs[['two_year_recid','age','priors_count','c_degree_F','female','length_of_stay',\
                'African-American','Asian','Caucasian','Hispanic','Native American']]
    df3=df3.copy()
    df3['conviction_more_than_2']=df3.priors_count.apply(lambda x: 1 if x>=2 else 0)
    X3=df3.loc[:,~df3.columns.isin(['two_year_recid'])]
    y3=df3.two_year_recid
    X3_train, X3_test, y3_train, y3_test= train_test_split(X3, y3, test_size=0.2, random_state=1)
    model=LogisticRegressionCV(max_iter=10000,cv=10,random_state=42,penalty='l2',solver='lbfgs').fit(X3_train,y3_train.values.ravel())
    return model




        # var = st.selectbox("Choose a Variable", cox_df.columns)
        # # histogram
        # fig = plt.figure()
        # plt.hist(cox_df[var])
        # plt.xticks(rotation=270)
        # st.pyplot(plt)