
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd


# In[19]:


df=pd.read_csv('Shenzhen_Clean.csv')


# In[20]:


df


# In[23]:


df=df.drop(df.index[456:463])

