#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from prediction import *


# In[10]:


model=jb.load(r'Model/model.joblib')
ordinal=jb.load(r'Model/ordinal_encoder.joblib')


# In[5]:


st.set_page_config(page_title='Accident Severity Prediction',layout='wide')


# In[6]:


#options
columns=["Number_of_vehicles_involved","Number_of_casualties","hour","Light_conditions","Age_band_of_driver","Day_of_week","Road_surface_conditions","Types_of_Junction","Driving_experience","Sex_of_casualty"]
cat_col=["light_condition","driver_age","day_of_week","road_surface_conditions","junction_type","driving_experience"]
days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
age=["18-30",'31-50','Over 51',"Unknown","Under 18"]
driver_exp=["5-10yr","2-5yr","Above 10yr","1-2yr",'Below 1yr',"No License","Unknown"]
light=["Daylight","Darkness-light lit","Darkness-no light","Darkness- lights unlit"]
vehicle_inv=20
casualties=20

road=["Dry","Wet or damp","Snow","Flood over 3cm deep"]
junction=["Y Shape","No Junction","Crossing","Other","Unknown","O Shape","T Shape","X Shape"]


# In[7]:


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)


# In[8]:


def main():
    with st.form('prediction form'):
        st.subheader("Enter the following features")
        v_inv=st.slider("Vehicle Involved: ",1,7,value=0,format="%d")
        cas=st.slider("Number of Casualties :",1,8,value=0,format="%d")
        hour=st.slider("Hour of Accident: ",1,24,value=1,format="%d")
        light_con=st.selectbox("Lightning Condiions: ",options=light)
        driver_age=st.selectbox("Driver Age: ",options=age)
        day=st.selectbox("Day of the Week: ",options=days)
        road_cond=st.selectbox("Road Surface Condition: ",options=road)
        junc=st.selectbox("Junction Type: ",options=junction)
        exp=st.selectbox("Driver Experience: ",options=driver_exp)
        sex_cas=st.radio("Sex Of Casualty",options=["Male","Female"])
        submit=st.form_submit_button("Predict")
        
        if submit:
            df=pd.DataFrame(np.array([v_inv,cas,hour-1,light_con,driver_age,day,road_cond,junc,exp,sex_cas]).reshape(1,-1),columns=columns)
            df["Number_of_vehicles_involved"]=df["Number_of_vehicles_involved"].astype(int)
            df["Number_of_casualties"]=df["Number_of_casualties"].astype(int)
            df["hour"]=df["hour"].astype(int)
            st.write(df)
            df=ordinal_encoder(df,ordinal)
            pred=predict(df,model)
            st.write(f"The predicted severity is:  {pred[0]}")
            
                                     


# In[9]:


if __name__ == '__main__':
    main()


# In[ ]:




