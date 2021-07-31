#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


to_train = pd.read_csv('to_train.csv')


# In[5]:


from sklearn.model_selection import train_test_split
training_set, testing_set = train_test_split(to_train, test_size=0.2)


# In[6]:


x_train = training_set['var1(t-1)']
y_train = training_set['var1(t)']
x_test = testing_set['var1(t-1)']
y_test = testing_set['var1(t)']


# In[7]:


X_train = np.array(x_train).reshape(-1,1)
Y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(x_test).reshape(-1,1)
Y_test = np.array(y_test).reshape(-1,1)


# In[8]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[10]:


# create and fit the LSTM network
deep_model = Sequential()
deep_model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
deep_model.add(Dense(1, kernel_initializer='normal'))
deep_model.compile(loss='mean_squared_error', optimizer='adam')


# In[11]:


deep_model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=2)


# In[12]:


# make predictions
trainPredict = deep_model.predict(X_train)
testPredict = deep_model.predict(X_test)


# In[24]:


# calculate root mean squared error
import math
from sklearn.metrics import mean_squared_error

trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


# In[ ]:




