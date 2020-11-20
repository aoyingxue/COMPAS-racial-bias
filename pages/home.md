# COMPAS Racial Bias Analysis and Score Prediction
## Data Description

COMPAS stands for Correctional Offender Management Profiling for Alternative Sanctions. Predictive analytics has entered into U.S. criminal justice system in the recent years. Judges and other court systems have been using machine learning algorithms to assess the risk of paroling an inmate, predicting the likelihood of becoming a recidivist, and accordingly making bailing or sentencing decisions. However, the accuracy is skeptical by the public. A model with only around 60% of accuracy as shown in the past studies could decide one's fate, which is not fair. If one could be aware of the high risk score he could probably get from the scoring system, he/she may find legal help in reaction to it beforehand. 

Data, provided by [Kaggle](https://www.kaggle.com/danofer/compass?select=cox-violent-parsed_filt.csv), contains variables used by the COMPAS algorithm in scoring defendants, along with their outcomes within 2 years of the decision, for over 10,000 criminal defendants in Broward County, Florida. Using the data, we're trying to identify the racial bias in inmate COMPAS reoffense risk scores.

2 different datasets are implemented in the analysis: 
- *"compas-scores-raw.csv"* which is the raw scores
- *"cox-violent-parsed.csv"* which is the Compass Score after some parsing and cleaning.

Due to the limited time, we picked the parsed score dataset as our main target of analysis, and we only picked some of the variables (**race, age, past criminal records, recidivism, charge degree, gender and age.**) out to test. Test results may be not the most accurate for now, but increasing the accuracy rate in scores would definitely be the future goal.

## About the Website
Mainly 3 parts would be presented in the website:
- Exploratory Data Analysis
- Model Building
- Score Prediction

## Variables
### The demographic variables:
- Id: unique key for each individual
- compas_screening_date: the screening date of COMPAS
- sex: sex (female, male)
- age: age (from 18 to 96)
- age_cat: age category (Greater than 45, 25 - 45, Less than 25)
- race: race ('Other', 'Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American')

### The variables indicating past criminal records:
- priors_count: number of priors
- charge_degree: including general recidivism and violent recidivism
    >Florida law divides crimes into felonies and misdemeanors. Felonies in Florida are punishable by death or incarceration in state prison and classified as capital or life felonies, or felonies of the first, second, or third degree. Misdemeanors are less serious crimes, punishable by up to one year in county jail. ([Florida Felony Crimes by Class and Sentences](https://www.criminaldefenselawyer.com/resources/criminal-defense/state-felony-laws/florida-felony-class.htm ))

### The outcome variables:
- decile_score: COMPAS scores of the risk of recidivism, ranging from 1 to 10, with 10 the highest risk. Scores 1-4 are labeled as "low", 5-7 "medium", 8-10 "high".


## Reference
Northpointe, Practitioners Guide to COMPAS, August 17, 2012, http://www.northpointeinc.com/files/technical_documents/FieldGuide2_081412.pdf
