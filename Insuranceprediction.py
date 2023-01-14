#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:/IPL assignment and PPT/Practise Session/Prediction of Insurance Charges/insurance.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.isnull().sum() #no missing values


# In[6]:


sns.countplot(data=df,x='sex')


# In[8]:


sns.countplot(data=df,x='smoker',hue='sex')


# In[9]:


sns.barplot(data=df,x='sex',y='charges')
plt.show()


# In[10]:


sns.barplot(data=df,x='smoker',y='charges')
plt.show()


# In[11]:


plt.hist(df.age,bins=100,color ='green')#People aged less than 20 are taking more insurances


# In[12]:


sns.scatterplot(data=df, x='age', y='charges',hue='smoker')#For the same age a person with the habit of smoking is charged more. 


# In[14]:


sns.scatterplot(data=df, x='bmi', y='charges',hue='smoker')#For the same bmi a person with the habit of smoking is charged more. 


# In[17]:


#Encode the column 'Smoker ' with 1 and 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categ = ['smoker','sex']
#df.categ = le.fit_transform(df.categ)
df[categ] = df[categ].apply(le.fit_transform)
#df.sex = le.fit_transform(df.sex)
df.head(10)


# In[20]:


df = pd.get_dummies(df, columns = ['region'])


# In[22]:


df.drop(['index'],axis=1,inplace=True)


# In[23]:


df=df.loc[:,['age','sex','bmi','children','smoker','region_northeast','region_northwest','region_southeast','region_southwest','charges']]


# In[24]:


df.head(10)


# In[25]:


df.corr()


# In[26]:


hm=sns.heatmap(df.corr(),vmin=0.1,vmax=0.9,annot=True)
#COulmns 'smoker' and 'bmi' has highest correlation with charges
#Coulmns 'sex' and 'children' are having less correlation with y varible


# In[35]:


#USe linear regression model for prediction
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[34]:


regressor_pred_score = regressor.score(X_test,y_test)
regressor_pred_score


# In[ ]:




