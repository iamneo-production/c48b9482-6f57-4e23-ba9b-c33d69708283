#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
df1=pd.read_csv('archive_1.csv')
l=[]
for d in df1['time']:
    l.append(d.split('T')[0])
df1['Date']=l

df1.drop(['time'],axis=1,inplace=True)
temp1=df1.groupby(['Date']).agg({'temperature_2m (°C)': ['min', 'max']})
temp2=df1.drop(['temperature_2m (°C)'],axis=1).groupby(['Date']).mean()
final_df1=temp1.join(temp2)
final_df1.reset_index(inplace=True)
df2=pd.read_excel('AQI_Data.xlsx',na_values='None')
df2.fillna(value='mean',inplace=True)
df2['Date']=final_df1['Date']
df2.set_index('Date',inplace=True)
final_df1.set_index('Date',inplace=True)
final=pd.concat([df2,final_df1],axis=1)

