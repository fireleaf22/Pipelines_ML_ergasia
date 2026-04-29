#import libraries
import streamlit as st
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#App title
st.title('ML App')

st.info('This is the App for the project')

#Read Dataset
df = pd.read_csv('https://raw.githubusercontent.com/fireleaf22/Pipelines_ML_ergasia/refs/heads/master/student-por.csv')

df_columns = list(df.columns)
#target_input = st.selectbox('select target',df_columns)
target_input = df.G3
target_input
##Data preparations
#Create sidebar for data input
with st.sidebar:
  st.header("Input features")
  sex = st.selectbox("sex",("F","M"))
  age = st.slider("age",15,22,18)
  famsize = st.selectbox("Family members (3+/3-)",("GT3","LE3"))
  Pstatus = st.selectbox("Parents (apart/together)",("A","T"))
  Medu = st.slider("mom's education",0,4,2)
  Fedu = st.slider("dad's education",0,4,2)
  studytime = st.slider("study time",1,4,2)
  failures = st.slider("failures",0,3,1)
  schoolsup = st.selectbox("school support",("yes","no"))
  famsup = st.selectbox("family support",("yes","no"))
  paid = st.selectbox("paid learning",("yes","no"))
  activities = st.selectbox("outside activities",("yes","no"))
  nursery = st.selectbox("nursery school",("yes","no"))
  higher = st.selectbox("higher education",("yes","no"))
  internet = st.selectbox("internet access",("yes","no"))
  romantic = st.selectbox("romantic",("yes","no"))
  freetime = st.slider("free time",1,5,3)
  goout = st.slider("goes out",1,5,3)
  health = st.slider("health",1,5,3)
  absences = st.slider("absences",0,32,10)
  G1 = st.slider("Grade1",0,19,10)
  G2 = st.slider("Grade2",0,19,10)
  G3 = st.slider("Grade3",0,19,10)

  #create dataframe for input
  data = {'sex': sex,
          'age': age,
          'famsize': famsize,
          'Pstatus': Pstatus,
          'Medu': Medu,
          'Fedu': Fedu,
          'studytime': studytime,
          'failures': failures,
          'schoolsup': schoolsup,
          'famsup': famsup,
          'paid': paid,
          'activities': activities,
          'nursery': nursery,
          'higher': higher,
          'internet': internet,
          'romantic': romantic,
          'freetime': freetime,
          'goout': goout,
          'health': health,
          'absences': absences,
          'G1': G1,
          'G2': G2,
          'G3': G3,
         }
  input_df = pd.DataFrame(data, index=[0])
  input_features = pd.concat([input_df,df], axis=0)

#Show input data
with st.expander('Input Data'):
  input_df
#input_features
##Info Show
#Show Dataset
with st.expander('Data'):
  st.write('**Raw Dataset**')
  df

  st.write('**Features**')
  Features = input_features.drop(target_input, axis=1)
  Features

  st.write('**Target**')
  target_input
  

#Show Charts for the data---------add more charts
with st.expander('Charts'):
  st.bar_chart(
    df,
    x="G3",
    y="studytime",
    color="sex",
    horizontal=True
)
  



##Encoding
#Encode Features
encode = ['sex', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid',
          'activities', 'nursery', 'higher', 'internet', 'romantic']
df_features = pd.get_dummies(input_features, prefix=encode)
encoded_freatures = df_features[1:]
input_row = df_features[:1]


#Encode Target (not needed at the moment)









#Model training
clf = RandomForestClassifier()
clf.fit(encoded_freatures, target_input) #encoded target if needed

#Apply model for predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)


#Predictions
with st.expander('Predictions'):
  st.write('**Probabilities**')
  prediction_proba
  st.write('**Final Prediction**')
  prediction


