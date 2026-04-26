import streamlit as st
import pandas as pd

st.title('ML App')

st.info('This is the App for the project')

with st.expander('Data'):
  st.write('**Raw Dataset**')
  df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/Video_Games_Sales_as_at_22_Dec_2016.csv')
  df


with st.expander('Chart'):
  #st.bar_chart(df, x="critic_score", y="total_sales", stack=False)
  #st.bar_chart(df, x="critic_score", y="total_sales", color="genre", horizontal=True)
  st.scatter_chart(
    df,
    x="Global_Sales",
    y="Critic_Score",
    color="Genre",
  )

#Data preperations
#with st.

