from turtle import width
import streamlit as st
import pandas as pd
import numpy as np

st.title('DEMO')

st.write('This is a demo of Streamlit')

df = pd.read_csv('result.csv')
df_DBSCAN = pd.read_csv('resultDBSCAN.csv')



st.write('This is a detail table of the result of the HILANDER algorithm')
st.dataframe(df.drop('test_path', axis=1))
st.write('This is a detail table of the result of the DBSCAN algorithm')
#st.dataframe(df_DBSCAN) 