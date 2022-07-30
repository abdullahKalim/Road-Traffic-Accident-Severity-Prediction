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
columns=["vehicles_involved","casualties","light_condition","driver_age","day_of_week","road_surface_conditions","junction_type","driving_experience"]
cat_col=["light_condition","driver_age","day_of_week","road_surface_conditions","junction_type","driving_experience"]
days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
age=["18-30",'31-50','Over 51',"Unknown","Under 18"]
driver_exp=["5-10yr","2-5yr","Above 10yr","1-2yr",'Below 1yr',"No License","Unknown"]
light=["Daylight","Darkness-light lit","Darkness-no light","Darkness- lights unlit"]
vehicle_inv=20
casualties=20

road=["Dry","Wet or damp","Snow","Flood over 3cm deep"]
junction=["Y shape","No Junction","Crossing","Other","Unknown","O shape","T shape","X shape"]


# In[7]:


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)


# In[8]:


def main():
    with st.form('prediction form'):
        st.subheader("Enter the following features")
        v_inv=st.slider("Vehicle Involved: ",0,vehicle_inv,value=0,format="%d")
        cas=st.slider("Number of Casualties :",0,20,value=0,format="%d")
        hour=st.slider("Hour of Accident: ",1,24)
        light_con=st.selectbox("Lightning Condiions: ",options=light)
        driver_age=st.selectbox("Driver Age: ",options=age)
        day=st.selectbox("Day of the Week: ",options=days)
        road_cond=st.selectbox("Road Surface Condition: ",options=road)
        junc=st.selectbox("Junction Type: ",options=junction)
        exp=st.selectbox("Driver Experience: ",options=driver_exp)
        sex_cas=st.radio("Sex Of Casualty",options=["Male","Female"])
        submit=st.form_submit_button("Predict")
        
        if submit:
            df=np.array([v_inv,cas,hour-1,light,driver_age,day,road_cond,junc,exp,sex_cas],columms=columns)
            df=ordinal_encoder(df,df.reshape(-1,1))
            data=np.array(df).reshape(1,-1)
            pred=predict(df,model)
            st.write(f"The predicted severity is:  {pred[0]}")
            
                                     


# In[9]:


if __name__ == '__main__':
    main()


# In[ ]:




