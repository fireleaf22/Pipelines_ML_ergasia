import streamlit as st
import pandas as pd

st.title('ML App')

st.info('This is the App for the project')

df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/Video_Games_Sales_Cleaned.csv')
df
