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

    ## first container: show dataset
    st.write("## View Raw Dataset")
    st.write(
        """
        If you'd like to see the raw dataset that we used to do the analysis, 
        please choose the dataset you'd like to see and check the box down below.
        """
    )
    show_dataset()
    st.write(
        """
        We used the parsed dataset as our main dataset to conduct our analysis, 
        using the variables presented at "Home" page. 
        We started our analysis by looking at the distribution of COMPAC decile scores. 
        """
    )

    ## 2nd container: show decile score distribution
    st.write("## Score Distribution")
    cox_df = preprocess_cox()
    ## general recidivism
    show_general_score_dstrbt(cox_df)
    st.write(
        """
        The histograms shown above are the distributions of the risk scores for general recidivism. 
        The difference between caucasian defendants and black defendants is very obvious.
        Scores for the caucasian defendants are skewed over the lower risk scores, 
        while scores for the black defendants are almost evenly distributed among different levels of risks.
        """
    )
    ## violent recidivism
    show_violent_score_dstrbt(cox_df)
    st.write(
        """
        The histgorams shown above are the distributions of the risk scores for violent recidivism.
        It also shows a clear difference between caucasian and black defendants in score distribution.
        """
    )
    ## female vs. male
    show_sex_score_dstrbt(cox_df)
    st.write(
        """
        We also tried to see if there's also the same kind of disparity among other factors, such as sex, 
        which may also be an important factor to identify when it comes to crime.
        The sample size of females is much smaller than males as expected.
        However, after we calculated the percentage distribution among each sex, 
        we found out that decile scores are distributed quite evenly for both sex. 
        No apparent disparity between two sexes.
        """
    )


def show_dataset():
    dataset_name = st.selectbox(
        "Select Dataset",
        ("COMPAS Raw Scores", "Compass Score after parsing and cleaning"),
    )
    show_data = load_data(100, dataset_name)
    # # checkbox
    if st.checkbox("View Chosen Data: %s" % dataset_name):
        st.subheader(dataset_name)
        st.dataframe(show_data)


def preprocess_cox():
    ## parse data
    df = pd.read_csv(
        "data/cox-violent-parsed.csv",
        parse_dates=[
            "dob",
            "compas_screening_date",
            "c_jail_in",
            "c_jail_out",
            "c_offense_date",
            "c_arrest_date",
            "r_offense_date",
            "vr_offense_date",
            "screening_date",
            "v_screening_date",
            "in_custody",
            "out_custody",
        ],
    )
    ## drop the duplicated columns for the same person
    df = df.copy()
    df = df.loc[:, ~df.columns.isin(["id", "start", "end"])]
    df.drop_duplicates(["name", "sex", "dob"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    ## to filter the columns needed
    cox_df = df[
        [
            "sex",
            "age",
            "race",
            "decile_score",
            "v_decile_score",
            "priors_count",  # number of priors
            "c_charge_degree",
            "is_recid",
            "r_charge_degree",  # general recidivism
            "is_violent_recid",
            "vr_charge_degree",  # violent recidivism
        ]
    ]
    ## turn sex, race into one-hot encoding
    cox_df = cox_df.copy()
    cox_df["female"] = cox_df.sex.apply(lambda x: 0 if x == "Male" else 1)
    cox_df = cox_df.loc[
        cox_df.decile_score != -1,
    ]
    return cox_df


def show_general_score_dstrbt(cox_df):
    with st.beta_container():
        st.write("### Decile Scores for General Recidivism")
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        plt.title("General Recidivism\n of Caucasian")
        plt.hist(
            cox_df.loc[cox_df.race == "Caucasian", "decile_score"],
            rwidth=0.75,
            bins=np.arange(1, 12) - 0.5,
        )
        plt.xticks(np.arange(1, 11))
        plt.yticks(np.arange(0, 1300, 200))

        ax2 = fig.add_subplot(122)
        plt.title("General Recidivism\n of Black")
        plt.hist(
            cox_df.loc[cox_df.race == "African-American", "decile_score"],
            rwidth=0.75,
            bins=np.arange(1, 12) - 0.5,
        )
        plt.xticks(np.arange(1, 11))
        plt.yticks(np.arange(0, 1300, 200))
        ## show the plot through streamlit
        st.pyplot(plt)

def show_violent_score_dstrbt(cox_df):
    with st.beta_container():
        st.write("### Decile Scores for Violent Recidivism")
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        plt.title("Decile Scores for Violent Recidivism of Caucasian Defendants")
        plt.hist(
            cox_df.loc[cox_df.race == "Caucasian", "v_decile_score"],
            rwidth=0.75,
            bins=np.arange(1, 12) - 0.5,
        )
        plt.xticks(np.arange(1, 11))
        plt.yticks(np.arange(0, 1700, 200))

        ax2 = fig.add_subplot(122)
        plt.title("Decile Scores for Violent Recidivism of Black Defendants")
        plt.hist(
            cox_df.loc[cox_df.race == "African-American", "v_decile_score"],
            rwidth=0.75,
            bins=np.arange(1, 12) - 0.5,
        )
        plt.xticks(np.arange(1, 11))
        plt.yticks(np.arange(0, 1700, 200))
        ## show the plot through streamlit
        st.pyplot(plt)

def show_sex_score_dstrbt(cox_df):
    with st.beta_container():
        st.write("### Decile Scores of General Recidivism for Females and Males")
        female=[0]*10
        male=[0]*10
        for i in list(range(1,11)):
            female[i-1]=cox_df.loc[(cox_df.female==1) & (cox_df['decile_score']==i),].shape[0]
            male[i-1]=cox_df.loc[(cox_df.female==0) & (cox_df['decile_score']==i),].shape[0]
        female=[f/sum(female) for f in female]
        male=[m/sum(male) for m in male]
        fig=plt.figure(figsize=(15,5))
        ax1=fig.add_subplot(121)
        plt.title("Decile Scores for General Recidivism of Female Defendants")
        plt.bar(height=female,x=range(1,11))
        plt.xticks(np.arange(1,11))
        plt.yticks(np.arange(0,1,0.1))

        ax2=fig.add_subplot(122)
        plt.title("Decile Scores for General Recidivism of Male Defendants")
        plt.bar(height=male,x=range(1,11))
        plt.xticks(np.arange(1,11))
        plt.yticks(np.arange(0,1,0.1))
        ## show the plot through streamlit
        st.pyplot(plt)