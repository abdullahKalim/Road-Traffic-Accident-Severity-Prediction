#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import joblib as jb
import pandas as pd



# In[8]:



# In[2]:


def ordinal_encoder(df,ordinal): 
    
    return ordinal.transform(df)


# In[ ]:


# In[9]:


def predict(data,model):
    return model.predict(data)
    


# In[ ]:




