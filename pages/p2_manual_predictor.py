!pip install scikit-learn
!pip install xgboost 
!pip install plotly

import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from sklearn.preprocessing import MinMaxScaler

st.header("Manual Predictor")
st.write("This section allows users to enter key agricultural parameters manually to generate real-time yield predictions. Users can input data such as soil quality metrics, weather conditions, and crop variety. These inputs are then processed through predictive model to estimate potential harvest outcomes.")

st.markdown("\n\n")

# crop selction 
crop_selection_opt = st.selectbox(
    "1. What type of crops do you choose? ",
    ("Tomato", "Sugar Cane", "Rice", "Corn", "Cotton","Others"),
)


# number input for Nitrogen/N
n_lvl = st.number_input("Input the N/Nitrogen level of the soil:", min_value = 0.0, value = 50.0)

# number input for phosphorous
p_lvl = st.number_input("Input the P/Phosphorous level in the soil::", min_value = 0.0, value = 50.0)


# numbver input for humidity
hum_lvl = st.number_input("Input the humidity level of the surrounding area:", min_value = 0.0, value = 80.0)

# number input for Temperature
temp = st.number_input("Input the the tempurature of the surrounding area:", min_value = 0.0, value = 25.0)

# number input for soil quality level
sq_lvl = st.number_input("Input the soil quality level:", min_value = 0.0, value = 70.0)

# 'Temperature', 'Humidity', 'P', 'N', 'Soil_Quality'
num_data = [[temp,hum_lvl,p_lvl,n_lvl,sq_lvl]]

#loading the scaler
import joblib
scaler = joblib.load('scaler.pkl')

# scaling the numerical data
col_names = ['Temperature', 'Humidity', 'P', 'N', 'Soil_Quality']
new_data_df = pd.DataFrame(num_data, columns=col_names)
num_data_scaled = scaler.transform(new_data_df)
num_data_scaled = pd.DataFrame(np.array(num_data_scaled).reshape(1,-1), columns=col_names)

# Crop type Selection 
col_names2 = ['Crop_Type_Tomato', 'Crop_Type_Sugarcane','Crop_Type_Rice','Crop_Type_Corn','Crop_Type_Cotton']
ct_dict = {
    "Tomato":[1,0,0,0,0],
    "Sugar Cane":[0,1,0,0,0],
    "Rice":[0,0,1,0,0],
    "Corn":[0,0,0,1,0],
    "Cotton":[0,0,0,0,1],
    "Others":[0,0,0,0,0]
        }

df_ct = pd.DataFrame([ct_dict[crop_selection_opt]], columns=col_names2)

# concating the two dataframe
df_cat = pd.concat([num_data_scaled, df_ct], axis=1)

df_cat['Crop_Type_Barley'] = 0 
df_cat['Crop_Type_Potato'] = 0
df_cat['Crop_Type_Soybean'] = 0
df_cat['Crop_Type_Sunflower'] = 0
df_cat['Crop_Type_Wheat'] = 0 
df_cat['Soil_pH'] = 0
df_cat['Wind_Speed'] = 0
df_cat['K'] = 0

# reordering the columns
cor_order = ['Soil_pH',
 'Temperature',
 'Humidity',
 'Wind_Speed',
 'N',
 'P',
 'K',
 'Soil_Quality',
 'Crop_Type_Barley',
 'Crop_Type_Corn',
 'Crop_Type_Cotton',
 'Crop_Type_Potato',
 'Crop_Type_Rice',
 'Crop_Type_Soybean',
 'Crop_Type_Sugarcane',
 'Crop_Type_Sunflower',
 'Crop_Type_Tomato',
 'Crop_Type_Wheat']

df_final = df_cat.loc[:, cor_order]
# st.write(df_final)

# importing models
import xgboost as xgb
xbg_reg = xgb.XGBRegressor()
xbg_reg.load_model("xgboost_model_scaled.json")
pred = xbg_reg.predict(df_final)

st.divider()
st.markdown("Under these conditions, the crop selected will yield **" + str(pred[0]) + " tons/hectare.**")
