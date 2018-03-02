
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd


# In[94]:


df=pd.read_csv('Shenzhen_Clean.csv')


# In[95]:


df


# In[96]:


df=df.drop(df.index[456:463])


# In[97]:


#Merge Date
#Feature:Kitchen waste (t) Fruit and vegetable waste (t) Bread Paste (t) Waste oil (t)


# In[98]:


df1=df[['Kitchen waste (t)','Fruit and vegetable waste (t)','Bread Paste (t)','Waste oil (t)','Total Waste (t)','Diesel waste water (m³)','Flour and waste oil (m³)','Kitchen waste paste (m³)','#1 acidification hydrolysis tank feed (m³)','#1 acidification hydrolysis tank discharge (m³)','#2 acidification hydrolysis tank feed (m³)','#2 acidification hydrolysis tank discharge (m³)','#1 Anaerobic tank slurry feed (m³)','#1 Anaerobic tank biogas cumulative production (m³)','#1 anaerobic tank biogas daily output (m³)','#2 Anaerobic tank slurry feed (m³)','#2 anaerobic tank biogas cumulative production (m³)','#2 anaerobic tank biogas daily output (m³)']]


# In[99]:


df1.head()


# In[100]:


df1['acid_feed']=df1['#1 acidification hydrolysis tank feed (m³)']+df1['#2 acidification hydrolysis tank feed (m³)']
df1['acid_discharge']=df1['#1 acidification hydrolysis tank discharge (m³)']+df1['#2 acidification hydrolysis tank discharge (m³)']
df1['anaerobic_feed']=df1['#1 Anaerobic tank slurry feed (m³)']+df1['#2 Anaerobic tank slurry feed (m³)']
df1['anaerobic_cumuprod']=df1['#1 Anaerobic tank biogas cumulative production (m³)']+df1['#2 anaerobic tank biogas cumulative production (m³)']
df1['anaerobic_dailyoutput']=df1['#1 anaerobic tank biogas daily output (m³)']+df1['#2 anaerobic tank biogas daily output (m³)']


# In[101]:


df1.columns = df1.columns.map(lambda x: x.replace(' ','_'))


# In[102]:


df1.columns = df1.columns.map(lambda x: x.replace('#',''))


# In[103]:


df1.columns = df1.columns.map(lambda x: x.replace('(t)',''))
df1.columns = df1.columns.map(lambda x: x.replace('(m³)',''))


# In[105]:


df1.to_csv('Shenzhen_useful.csv')

