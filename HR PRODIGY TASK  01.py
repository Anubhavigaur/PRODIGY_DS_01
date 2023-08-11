#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the train dataset
train = pd.read_csv('train.csv')


# In[3]:


#inspect the first few rows of the train dataset
display(train.head())


# In[4]:


# set the index to passengerId
train = train.set_index('PassengerId')


# In[5]:


#load the test dataset
test = pd.read_csv('test.csv')


# In[6]:


#inspect the first few rows of the test dataset
display(test.head())


# In[7]:


#By calling the shape attribute of the train dataset we can observe that there are 891 observations and 11 columns.

train.shape


# In[8]:


# Check out the data summary
# We have analysis the missing data given in the dataset
train.head()


# In[9]:


# identify datatypes of the train dataset with carrying a variable named as datatype.
datatype= pd.DataFrame(train.dtypes)
datatype


# In[10]:


# We have analysis the null or 0 values in the dataset with the help of datatype.
datatype['MissingVal'] = train.isnull().sum()
datatype


# In[11]:


# Identify number of unique values, For object nunique will the number of levels
# Add the stats the data type
datatype['NUnique']=train.nunique()
datatype
# We have analayis the given number of unique values in the dataset.


# In[12]:


# Identify the count for each variable, add the stats to datadict
datatype['Count']=train.count()
datatype


# In[13]:


# rename the 0 column
datatype = datatype.rename(columns={0:'DataType'})
datatype


# In[14]:


# get discripte statistcs on "object" datatypes
train.describe(include=['object'])


# In[15]:


# get discriptive statistcs on "number" datatypes
train.describe(include=['number'])


# In[16]:


train.Survived.value_counts(normalize=True)


# In[17]:


fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train,ax=axes[0,0])
sns.countplot('Pclass',data=train,ax=axes[0,1])
sns.countplot('Sex',data=train,ax=axes[0,2])
sns.countplot('SibSp',data=train,ax=axes[0,3])
sns.countplot('Parch',data=train,ax=axes[1,0])
sns.countplot('Embarked',data=train,ax=axes[1,1])
sns.distplot(train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train['Age'].dropna(),kde=True,ax=axes[1,3])


# In[18]:


figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=train,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train,ax=axesbi[1,2])


# In[19]:


sns.jointplot(x="Age", y="Fare", data=train);


# In[21]:


# we see that there are 15 Zero values and its reasonbale 
# to flag them as missing values since every ticket 
# should have a value greater than 0
print((train.Fare == 0).sum())


# In[22]:


# mark zero values as missing or NaN
train.Fare = train.Fare.replace(0, np.NaN)


# In[23]:


# validate to see if there are no more zero values
print((train.Fare == 0).sum())


# In[24]:


train[train.Fare.isnull()].index


# In[25]:


train.Fare.mean()


# In[26]:


train.Fare.fillna(train.Fare.mean(),inplace=True)


# In[27]:


# validate if any null values are present after the imputation
train[train.Fare.isnull()]


# In[28]:


# we see that there are 0 Zero values
print((train.Age == 0).sum())


# In[29]:


# impute the missing Age values with the mean Fare value
train.Age.fillna(train.Age.mean(),inplace=True)


# In[30]:


# validate if any null values are present after the imputation
train[train.Age.isnull()]


# In[31]:


train.Cabin.isnull().mean()


# In[32]:


train.info()


# In[33]:


train.columns


# In[35]:


trainL = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked']]


# In[36]:


trainL = trainL.dropna()


# In[37]:


trainL.isnull().sum()


# In[38]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[39]:


# Regression on survival on Age
X_Age = trainL[['Age']].values
y = trainL['Survived'].values
# Use the fit method to train
lr.fit(X_Age,y)
# Make a prediction
y_predict = lr.predict(X_Age)
y_predict[:10]
(y == y_predict).mean()


# In[40]:


# Regression on survival on Fare
X_Fare = trainL[['Fare']].values
y = trainL['Survived'].values
# Use the fit method to train
lr.fit(X_Fare,y)
# Make a prediction
y_predict = lr.predict(X_Fare)
y_predict[:10]
(y == y_predict).mean()


# In[41]:


# Regression on survive on Sex(using a Categorical Variable)
X_sex = pd.get_dummies(trainL['Sex']).values
y = trainL['Survived'].values
# Use the fit method to train
lr.fit(X_sex, y)
# Make a prediction
y_predict = lr.predict(X_sex)
y_predict[:10]
(y == y_predict).mean()


# In[42]:


# Regression on survive on PClass(using a Categorical Variable)
X_pclass = pd.get_dummies(trainL['Pclass']).values
y = trainL['Survived'].values
lr = LogisticRegression()
lr.fit(X_pclass, y)
# Make a prediction
y_predict = lr.predict(X_pclass)
y_predict[:10]
(y == y_predict).mean()


# In[ ]:




