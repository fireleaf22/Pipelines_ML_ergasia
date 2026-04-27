import streamlit as st
import pandas as pd

st.title('ML App')

st.info('This is the App for the project')

with st.expander('Data'):
  st.write('**Raw Dataset**')
  df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/student-por.csv')
  df

  st.write('**X**')
  X = df.drop('G3', axis=1)
  X

  st.write('**Y**')
  Y = df.G3
  Y


with st.expander('Chart'):
  st.bar_chart(
    df,
    x="G3",
    y="studytime",
    color="sex",
    horizontal=True
)
  

#Data preparations
with st.sidebar:
  st.header('Input features')
  sex = st.selectbox('sex'('F','M'))
  age = st.slider('age',15,22,18)
  famsize = st.selectbox('Family members'('GT3','LE3'))
  Pstatus = st.selectbox('Parents: together?'('A','T'))
  Medu = st.slider('mom edu',0,4,2)
  Fedu = st.slider('dad edu',0,4,2)
  studytime = st.slider('study time',1,4,2)
  failures = st.slider('failures',0,3,1)
  schoolsup = st.selectbox('school support'('yes','no'))
  famsup = st.selectbox('family support'('yes','no'))
  paid = st.selectbox('paid learning'('yes','no'))
  activities = st.selectbox('activities'('yes','no'))
  nursery = st.selectbox('nursery school'('yes','no'))
  higher = st.selectbox('higher education'('yes','no'))
  internet = st.selectbox('internet'('yes','no'))
  romantic = st.selectbox('romantic'('yes','no'))
  famrel = st.slider('famrel',1,5,3)
  freetime = st.slider('free time',1,5,3)
  goout = st.slider('go out',1,5,3)
  health = st.slider('health',1,5,3)
  absences = st.slider('absences',0,32,10)
  G1 = st.slider('Grade1',0,20,10)
  G2 = st.slider('Grade2',0,20,10)
  G3 = st.slider('Grade3',0,20,10)


