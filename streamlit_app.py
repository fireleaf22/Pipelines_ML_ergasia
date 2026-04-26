import streamlit as st
import pandas as pd

st.title('ML App')

st.info('This is the App for the project')

with st.expander('Data'):
  st.write('**Raw Dataset**')
  df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/student-por.csv')
  df


with st.expander('Chart'):
  st.line_chart(
    df,
    x="G3",
    y="studytime",
    color="sex",
)
  

