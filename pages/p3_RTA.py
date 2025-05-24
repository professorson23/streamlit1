import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
import plotly.express as px  # interactive charts
import joblib

df = pd.read_csv("predicted_yield.csv")

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

st.title("Real-Time Dashboard")
st.write("The predicted output of the each crops will be displayed and updated daily based on the conditions provided. (**Multiple crops may be selected**)")
st.caption("The demo below uses synthetic data for its prediction.")

# crop select
crop_filter = st.multiselect(
    "Select the type of crops",
    df['Crop_Type'].unique()
)

filtered_df = df[df['Crop_Type'].isin(crop_filter)][['Date','Crop_Type','Final_Yield']]
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

today = pd.Timestamp.today().normalize()
start_date = today - pd.Timedelta(days=30)
df_recent = filtered_df[filtered_df['Date'].between(start_date, today)]

df_recent["Final_Yield"] = df_recent["Final_Yield"].clip(0)

fig = px.line(df_recent, x='Date', y='Final_Yield', color='Crop_Type',
              title="Crop Yield Over Time",
              markers=True)

fig.update_layout(xaxis_title="Date", yaxis_title="Yield (tonnes/hectare)")
st.plotly_chart(fig, use_container_width=True)