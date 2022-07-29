#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import joblib as jb
import pandas as pd


# In[7]:


min = list(range(5,56, 5))
def convert_minutes(x: int):
    for m in min:
        if x % m == x and x > m-5:
            return m
        if x in [56,57,58,59]:
            return 0
        if x in min+[0]:
            return x


# In[8]:



# In[2]:


def ordinal_encoder(df,ordinal): 
    
    return ordinal.tranform(df)


# In[ ]:


# In[9]:


def predict(data,model):
    return model.predict(data)
    


# In[ ]:




