#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.write("""
# Sales Prediction App
This app predicst the Sales of Rossman Pharmaceuticals
""")
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](to_train.csv)
""")
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Date = st.sidebar.date_input('Date')
        Store_Id = st.sidebar.number_input('Store_Id', min_value=0, max_value=10, step=1)
        DayOfWeek = st.sidebar.number_input('Day of Week', min_value=0, max_value=7,step=1)
        IsHoliday = st.sidebar.number_input('IsHoliday', min_value=0, max_value=1,step=1)
        IsWeekend = st.sidebar.number_input('IsWeekend', min_value=0, max_value=1,step=1)
        IsPromo = st.sidebar.number_input('IsPromo', min_value=0, max_value=1, step=1)
                                          
        data = {'Date': Date,
                'Store_Id': Store_Id,
                'DayOfWeek' : DayOfWeek,
                'IsHoliday': IsHoliday,
                'IsWeekend': IsWeekend,
                'IsPromo': IsPromo}
                                          
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
df = pd.DataFrame(input_df)
date_store = df.iloc[:, 0:2]
trained_df = df.drop(['Date', 'Store_Id'], axis=1)
    
     # Displays the user input features
st.subheader('User Input features')
    
    
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df) 
    
 # Reads in saved Regression model
load_reg = pickle.load(open('sales_reg.pkl', 'rb'))

# Apply model to make predictions
prediction = load_reg.predict(trained_df)
sales = pd.DataFrame(prediction)

#Shows new prediction
predicted_sales = pd.concat([date_store, trained_df, sales], axis=1)

st.subheader('Prediction')
st.write(predicted_sales)

def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href

st.markdown(get_table_download_link_csv(predicted_sales), unsafe_allow_html=True)




