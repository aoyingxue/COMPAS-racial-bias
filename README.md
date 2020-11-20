# About COMPAS Racial Bias
 Azure Hackathon: COMPAS Racial Bias and Predict Scores

## Authors
### Yuki (Yingxue) Ao (She/Her)
Responsible for the implementation of the whole project: Python-based web application, back-end deployment on Microsoft Azure, Building the model and predicting the scores. 
She is currently enrolled in Master of Science in Customer Analytics (MSCA) in Washington University in St. Louis. Actively in job hunting. Feel free to connect at [LinkedIn](https://www.linkedin.com/in/aoyingxue/) or [GitHub](https://github.com/aoyingxue). 

## Packages and Platforms
1. Python 3.5
2. Streamlit for providing the interactive web app template.
3. Google Colab Notebooks for building the models and predicting the scores.
4. Flask and Microsoft Azure to deploy the model and website.

## Installation
### Create virtual environment 
If running on Python 3, the `venv` is built-in:
`python3 -m venv env`
and then activate the environment:
`source env/bin/activate`
Install dependencies (details shown below).
`pip3 install -r requirements.txt`
### Run streamlit app
`streamlit run app.py`

## Dependencies
### Add or remove dependencies
We use *"pipreqs"* to generate *requirements.txt* which contains the required requirements for the deployment.

Add/remove dependencies at any point by updating requirements.txt (Python deps) or packages.txt (Debian deps) and doing a git push to your remote repo. This will cause Streamlit to detect there was a change in its dependencies, which will automatically trigger its installation.

It is best practice to pin Streamlit version in requirements.txt. Otherwise, the version may be auto-upgraded at any point without your knowledge, which could lead to undesired results (e.g. when we deprecate a feature in Streamlit).
