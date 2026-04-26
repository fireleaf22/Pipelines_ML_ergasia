import streamlit as st
import pandas as pd

st.title('ML App')

st.info('This is the App for the project')

with st.expander('Data'):
  st.write('**Raw Dataset**')
  df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/reels_attention_span_dataset_12000.csv')
  df


with st.expander('Chart'):
  st.area_chart(df, x="task_completion_rate", y="reels_watch_time_hours", color="stress_level")
  

