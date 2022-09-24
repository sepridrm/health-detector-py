import streamlit as st
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe
# from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
import math

# Remove whitespace from the top of the page and sidebar
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.css-hxt7ib{padding-top:1rem;padding-bottom:3rem;}</style>', unsafe_allow_html=True)

# Remove burger option
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write("""
# Health Detector
This app predicts health using **Naive Bayes**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tempf = st.sidebar.slider('Temperature (°C)', 33, 42, 36)
    pulse = st.sidebar.slider('Heart rate/minutes', 50, 130, 70)
    resp = st.sidebar.slider('Respiration/minutes', 10, 35, 15)
    sist = st.sidebar.slider('Sistolik', 80, 130, 90)
    dias = st.sidebar.slider('Diastolik', 50, 90, 60)
    sart = st.sidebar.slider('Saturasi', 80, 110, 95)
    # (temp*9/5) + 32,
    data = {'TEMPF': (tempf*9/5) + 32,
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
st.subheader('Total data: ' + str(len(df)))

cleaning = df.loc[(df["TEMPF"] > 95) & (df["TEMPF"] < 104)]
# st.write('Cleaning TEMPF: ' + str(len(cleaning)))
cleaning = cleaning.loc[(cleaning["PULSE"] > 50) & (cleaning["PULSE"] < 120)]
# st.write('Cleaning PULSE: ' + str(len(cleaning)))
cleaning = cleaning.loc[(cleaning["RESPR"] > 10) & (cleaning["RESPR"] < 20)]
# st.write('Cleaning RESPR: ' + str(len(cleaning)))
cleaning = cleaning.loc[(cleaning["BPSYS"] > 90) & (cleaning["BPSYS"] < 150)]
# st.write('Cleaning BPSYS: ' + str(len(cleaning)))
cleaning = cleaning.loc[(cleaning["BPDIAS"] > 60) & (cleaning["BPDIAS"] < 95)]
# st.write('Cleaning BPDIAS: ' + str(len(cleaning)))
cleaning = cleaning.loc[(cleaning["POPCT"] > 89) & (cleaning["POPCT"] < 100)]
# st.write('Cleaning POPCT: ' + str(len(cleaning)))
cleaning = cleaning.drop_duplicates()
# st.write('Cleaning Duplicate: ' + str(len(cleaning)))

# setting dataset
xtarget = cleaning.drop(cleaning.columns[6], axis=1)
ytarget = cleaning[cleaning.columns[6]]
# .head(1000)

# st.write(xtarget)
# st.write(ytarget)

xtrain, xtest, ytrain, ytest = train_test_split(xtarget, ytarget, test_size=0.2, random_state=42)

# st.write(selector(dtype_exclude="category")(xtrain))

#preproses
preproses = ColumnTransformer(
    transformers=[
        ('num', num_pipe(), ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"])
    ]
)

# using Naive Bayes Algorithm
pipeline = Pipeline([
    ('prep', preproses),
    ('algo', GaussianNB())
])

# training data
pipeline.fit(xtrain, ytrain)

st.subheader('Total data after Cleaning: ' + str(len(xtarget)))
st.subheader('Data training : ' + str(len(xtrain)) + ' with accuracy : ' + str(math.ceil(pipeline.score(xtrain, ytrain)*100)) + '%')
# st.subheader('Data testing : ' + str(len(xtest)) + ' with accuracy : ' + str(math.ceil(pipeline.score(xtest, ytest)*100)) + '%')

st.subheader('User Input parameters')
st.write(iu)

sd = score_define()
st.subheader('Class labels and their corresponding index number')
st.write(sd)

# prediction
st.subheader('Prediction Probability')
prediction = pipeline.predict(iu)
if prediction == 0:
    st.header('"Sehat"')
elif prediction == 1:
    st.header('"Cukup Sehat"')
elif prediction == 2:
    st.header('"Kurang Sehat"')
else:
    st.header('"Tidak Sehat"')