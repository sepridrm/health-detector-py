import streamlit as st
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import math

# Remove whitespace from the top of the page and sidebar
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.css-hxt7ib{padding-top:1rem;padding-bottom:3rem;}</style>', unsafe_allow_html=True)

st.write("""
# Health Detector
This app predicts health using **Naive Bayes**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    temp = st.sidebar.slider('Temperature (°C)', 33.0, 42.0, 36.5)
    pulse = st.sidebar.slider('Heart rate/minutes', 50, 130, 70)
    resp = st.sidebar.slider('Respiration/minutes', 10, 35, 15)
    sist = st.sidebar.slider('Sistolik', 80, 130, 90)
    dias = st.sidebar.slider('Diastolik', 50, 90, 60)
    sart = st.sidebar.slider('Sarutasi', 80, 110, 95)
    data = {'TEMPF': (temp*9/5) + 32,
            'PULSE': pulse,
            'RESPR': resp,
            'BPSYS': sist,
            'BPDIAS': dias,
            'POPCT': sart}
    features = pd.DataFrame(data, index=[0])
    return features

def score_define():
    data = {
        '0': 'Sehat',
        '1': 'Cukup Sehat',
        '2': 'Kurang Sehat',
        '3': 'Tidak Sehat',
    }
    features = pd.DataFrame(data, index=[0])
    return features

iu = user_input_features()

# import dataset from csv
filecsv = 'ehr.csv'
df = pd.read_csv(filecsv, header=0, delimiter=',', encoding='utf-8')

# setting dataset
xtarget = df.drop(df.columns[6], axis=1)
ytarget = df[df.columns[6]]
xtrain, xtest, ytrain, ytest = train_test_split(xtarget, ytarget, test_size=0.2, random_state=42)

# using Naive Bayes Algorithm
pipeline = Pipeline([
    ('algo', GaussianNB())
])
# training data
pipeline.fit(xtrain, ytrain)

st.subheader('Data training : ' + str(len(xtrain)) + ' with Training accuracy : ' + str(math.ceil(pipeline.score(xtrain, ytrain)*100)) + '%')

st.subheader('User Input parameters')
st.write(iu)

sd = score_define()
st.subheader('Class labels and their corresponding index number')
st.write(sd)

# prediction
st.subheader('Prediction Probability')
predictioon = pipeline.predict(iu)
if predictioon == 0:
    st.header('"Sehat"')
elif predictioon == 1:
    st.header('"Cukup Sehat"')
elif predictioon == 2:
    st.header('"Kurang Sehat"')
else:
    st.header('"Tidak Sehat"')