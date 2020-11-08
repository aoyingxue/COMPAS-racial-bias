# COMPAS-racial-bias
 Azure Hackathon: COMPAS Racial Bias and Predict Scores

## Authors
Python-based web application, and back-end deployment on Microsoft Azure are done by Yuki Ao (She/Her), Master of Science in Customer Analytics (MSCA) in Washington University in St. Louis. Feel free to connect at [LinkedIn](https://www.linkedin.com/in/aoyingxue/) or [GitHub](https://github.com/aoyingxue). 

Karlie Fang, MSCA in WashU, is responsible for Model building in Jupyter Notebook. 

## Packages and Platforms
1. Streamlit and Awesome Streamlit for providing the template for the web app visualizer.
2. Flask 
3. Microsoft Azure 

## Installation
### Create virtual environment 
If running on Python 3, the `venv` is built-in:
`python3 -m venv env`
and then activate the environment:
`source env/bin/activate`

## Dependencies
### Add or remove dependencies
We use *"pipreqs"* to generate *requirements.txt* which contains the required requirements for the deployment.

Add/remove dependencies at any point by updating requirements.txt (Python deps) or packages.txt (Debian deps) and doing a git push to your remote repo. This will cause Streamlit to detect there was a change in its dependencies, which will automatically trigger its installation.

It is best practice to pin Streamlit version in requirements.txt. Otherwise, the version may be auto-upgraded at any point without your knowledge, which could lead to undesired results (e.g. when we deprecate a feature in Streamlit).
