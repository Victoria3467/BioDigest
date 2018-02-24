
# coding: utf-8

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style='white', context='notebook', palette='deep')
plt.rcParams[ 'figure.figsize' ] = 9 , 5


# In[63]:


xl = pd.ExcelFile("HainanData_kindofClean.xlsx")


# In[64]:


df = xl.parse("Clean")


# In[65]:


df.shape


# In[66]:


nan_rows = df[df['day #2'].isnull()]


# In[67]:


nan_rows


# In[68]:


df.drop(['Day', 'Year', 'Water/m3', 'Total electricity cons (kWh)', '50% NaOH/kg', 'FeCl2/kg', 'PAM/kg', 'Defoamer/kg', 'day #2'], axis = 1, inplace = True)
df.head()


# In[62]:


df.describe()


# In[48]:


df.info()


# In[50]:


df[['Month', 'BioCNG Sold (m3)']].groupby(['Month']).mean()


# In[55]:


#sns.countplot(x='Month', hue="BioCNG Sold (m3)", data=df);

