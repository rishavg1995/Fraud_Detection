
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('claims.csv')


# In[3]:


data.head()


# In[4]:


data.keys()


# In[5]:


data.info()


# # Missing Data
# We used seaborn to create a simple heatmap to see where we are missing data!

# In[6]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Imbalanced Data

# In[7]:


sns.set_style('whitegrid')
sns.countplot(x = 'FraudFound_P', data = data, palette = 'RdBu_r')


# # Male vs Female

# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='Sex',data=data ,palette='RdBu_r')


# # Vehicle Category

# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='VehicleCategory',data=data ,palette='rainbow')


# # Accident Area

# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='AccidentArea',data=data ,palette='RdBu_r')


# # Driver Rating

# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='DriverRating',data=data ,palette='rainbow')


# # Age of Vehicle

# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='AgeOfVehicle',data=data ,palette='rainbow')


# # Police Report Filed

# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='PoliceReportFiled',data=data,palette='rainbow')


# # Witness

# In[14]:


sns.set_style('whitegrid')
sns.countplot(x='FraudFound_P',hue='WitnessPresent',data=data ,palette='rainbow')

