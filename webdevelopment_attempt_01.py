import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBRegressor

import streamlit as st
st.set_page_config(layout="wide")
with st.container():
    st.write("# VEHICLE PRICE PREDICTION")
data = pd.read_csv("data_ml.csv")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        dealer_name = st.selectbox("Dealer Name", options=data["dealer_name"].unique().tolist())
    with col2:
        export_status = st.selectbox("Export Status", options=data[data["dealer_name"]==dealer_name]["export_status"].unique().tolist())
    with col3:
        location = st.selectbox("Location", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)]["location"].unique().tolist())

def user_input_featutres():
    vehicle_name = st.sidebar.selectbox("Vehicle Name",options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)]["vehicle_name"].unique().tolist(),)
    model = st.sidebar.selectbox("Model", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)&(data["vehicle_name"]==vehicle_name)]["model"].unique().tolist())
    vehicle_type = st.sidebar.selectbox("Vehicle Type", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)&(data["vehicle_name"]==vehicle_name)&(data["model"]==model)]["vehicle_type"].unique().tolist())
    model_year = st.sidebar.selectbox("Vehicle Type", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)&(data["vehicle_name"]==vehicle_name)&(data["model"]==model)&(data["vehicle_type"]==vehicle_type)]["model_year"].unique().tolist())
    fuel_type = st.sidebar.selectbox("Fuel Type", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)&(data["vehicle_name"]==vehicle_name)&(data["model"]==model)&(data["vehicle_type"]==vehicle_type)&(data["model_year"]==model_year)]["fuel_type"].unique().tolist())
    spec = st.sidebar.selectbox("Spec", options=data[(data["dealer_name"]==dealer_name)&(data["export_status"]==export_status)&(data["location"]==location)&(data["vehicle_name"]==vehicle_name)&(data["model"]==model)&(data["vehicle_type"]==vehicle_type)&(data["model_year"]==model_year)&(data["fuel_type"]==fuel_type)]["spec"].unique().tolist())
    mileage = st.sidebar.selectbox("Mileage", options=range(0,101))
    input_data = {
        "vehicle_name":vehicle_name,
        "model":model,
        "vehicle_type":vehicle_type,
        "export_status":export_status,
        "dealer_name":dealer_name,
        "model_year":model_year,
        "location":location,
        "fuel_type":fuel_type,
        "spec":spec,
        "mileage":mileage
    }
    features = pd.DataFrame(input_data, index=[0])
    return features
df = user_input_featutres()
with st.container():
    st.subheader("Vehicle Input Features")
    st.dataframe(df)

data["model_year"] = data["model_year"].astype("category")
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = data.select_dtypes(include=["int", "float"]).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(data[cat_cols])
enc_cols = list(encoder.get_feature_names_out(cat_cols))
data[enc_cols] = encoder.transform(data[cat_cols])

train_ = data[enc_cols+num_cols]

min_max_sc = MinMaxScaler()
train_["mileage"] = min_max_sc.fit_transform(np.array(train_["mileage"]).reshape(-1,1))
train_["price"] = np.log(train_["price"])

input_cols = train_.columns.tolist()[:-1]
target_col = "price"

inputs = train_[input_cols]
targets = train_[target_col]

@st.cache(hash_funcs={'xgboost.sklearn.XGBRegressor': id})
def load_model_fit():
    xgb = XGBRegressor(n_jobs=-1, random_state=10, n_estimators=1000, max_depth=30)
    xgb.fit(inputs, targets)
    return xgb

def single_input_prediction(df):
    input_sample = pd.DataFrame(df, index=[0])
    input_sample["mileage"] = min_max_sc.transform(np.array(input_sample["mileage"]).reshape(-1,1))
    input_sample[enc_cols] = encoder.transform(input_sample[cat_cols])
    inputs = input_sample[enc_cols+["mileage"]]
    price_predicted = load_model_fit().predict(inputs)
    return np.ceil(np.exp(price_predicted[0]))

with st.container():
    st.subheader("Predicted Price of Vehicle")
    st.markdown('<p style=“font-size:300px;color:red”>' + "AED " + str(single_input_prediction(df)) + '</p>',
                unsafe_allow_html=True)

with st.container():
    st.subheader("Actual Price Details")
    # selected_data = data[(data["vehicle_name"] == df.loc[0, "vehicle_name"]) & (data["model"] == df.loc[0, "model"]) &
    #                 (data["dealer_name"] == df.loc[0, "dealer_name"])].reset_index(drop=True)
    actual_price = ["AED " + x for x in data[(data["vehicle_name"]==df.loc[0,"vehicle_name"])&(data["model"]==df.loc[0,"model"])&
                  (data["vehicle_type"]==df.loc[0,"vehicle_type"])&(data["export_status"]==df.loc[0,"export_status"])&
                  (data["dealer_name"]==df.loc[0,"dealer_name"])&(data["model_year"]==df.loc[0,"model_year"])&(data["location"]==df.loc[0,"location"])&
                  (data["fuel_type"]==df.loc[0,"fuel_type"])&(data["mileage"]==df.loc[0,"mileage"])&(data["spec"]==df.loc[0,"spec"])]["price"].tolist().]
    # record = pd.concat([selected_data,actual_price], axis=1)
    st.write(actual_price)
           
