#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('pip install dll')


# In[17]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import dill

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

to_train = pd.read_csv('to_train.csv')

from sklearn.model_selection import train_test_split
training_set, testing_set = train_test_split(to_train, test_size=0.2)

x_train = training_set['var1(t-1)']
y_train = training_set['var1(t)']
x_test = testing_set['var1(t-1)']
y_test = testing_set['var1(t)']


X_train = np.array(x_train).reshape(-1,1)
Y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(x_test).reshape(-1,1)
Y_test = np.array(y_test).reshape(-1,1)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# create and fit the LSTM network
deep_model = Sequential()
deep_model.add(Dense(3, kernel_initializer='normal'))
deep_model.compile(loss='mean_squared_error', optimizer='adam')

deep_model.fit(X_train, Y_train, epochs=3, batch_size=2, verbose=2)


dill.dump(deep_model, open('regression.pkl', 'wb'))


# In[ ]:





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


# In[1]:


import pickle
pickle.dump(deep_model, open('regression.pkl', 'wb'))


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


# In[43]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import weakref
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v1'

mlflow.set_experiment('Modelling')

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v1 = to_train.iloc[0:300000, :]
    
    # log parameters
    mlflow.log_param('data_url', v1)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', v1.shape[0])
    mlflow.log_param('input_cols', v1.shape[1])

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # create and fit the LSTM network
    deep_model = Sequential()
    deep_model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
    deep_model.add(Dense(1, kernel_initializer='normal'))
    deep_model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model on the training data
    deep_model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=2)

    # make predictions
    trainPredict = deep_model.predict(X_train)
    testPredict = deep_model.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.tensorflow.log_model(deep_model)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v2'

mlflow.set_experiment('NewModel')

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v2 = to_train.iloc[0:450000, :]
    
    # log parameters
    mlflow.log_param('data_url', v2)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', v2.shape[0])
    mlflow.log_param('input_cols', v2.shape[1])

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # create and fit the LSTM network
    deep_model = Sequential()
    deep_model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
    deep_model.add(Dense(1, kernel_initializer='normal'))
    deep_model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model on the training data
    deep_model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=2)

    # make predictions
    trainPredict = deep_model.predict(X_train)
    testPredict = deep_model.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.pyfunc.save_model(deep_model, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[58]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v2'

mlflow.set_experiment('InitialModel')
mlflow.start_run(nested = True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v2 = to_train.iloc[0:450000, :]

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from sklearn.linear_model import LinearRegression

    # create and fit the LSTM network
    model_1 = LinearRegression()
    
    # fit model on the training data
    model_1.fit(X_train, Y_train)

    # make predictions
    trainPredict = model_1.predict(X_train)
    testPredict = model_1.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.sklearn.save_model(model_1, "InitialModel1")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[62]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v2'

mlflow.set_experiment('Initial_Model')
mlflow.start_run(nested = True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v2 = to_train.iloc[0:450000, :]

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from sklearn.linear_model import LinearRegression

    # create and fit the LSTM network
    model_ = LinearRegression()
    
    # fit model on the training data
    model_.fit(X_train, Y_train)

    # make predictions
    trainPredict = model_.predict(X_train)
    testPredict = model_.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.sklearn.save_model(model_, "Initial_Model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[63]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v2'

mlflow.set_experiment('Initial_Model')
mlflow.start_run(nested = True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v2 = to_train.iloc[0:450000, :]

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from sklearn.tree import DecisionTreeRegressor

    # create and fit the LSTM network
    model_1 = DecisionTreeRegressor()
    
    # fit model on the training data
    model_1.fit(X_train, Y_train)

    # make predictions
    trainPredict = model_1.predict(X_train)
    testPredict = model_1.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.sklearn.save_model(model_1, "Initial_Model2")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[66]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v2'

mlflow.set_experiment('Initial_Model')
mlflow.start_run(nested = True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v2 = to_train.iloc[0:500000, :]

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from xgboost import XGBRegressor

    # create and fit the LSTM network
    model_2 = XGBRegressor()
    
    # fit model on the training data
    model_2.fit(X_train, Y_train)

    # make predictions
    trainPredict = model_2.predict(X_train)
    testPredict = model_2.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.sklearn.save_model(model_2, "Initial_Model3")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[47]:


v1 = to_train.iloc[0:400000, :]
v2 = to_train.iloc[0:450000, :]
v3 = to_train.iloc[0:50000, :]


# In[67]:


v1.to_csv('Version1.csv')
v2.to_csv('Version2.csv')
v3.to_csv('Version3.csv')


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import mlflow
import mlflow.sklearn

# data analysis and wrangling
import pandas as pd
import numpy as np

version = 'v3'

mlflow.set_experiment('Modelling')

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    v3 = to_train.iloc[0:50000, :]
    
    # log parameters
    mlflow.log_param('data_url', v3)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', v3.shape[0])
    mlflow.log_param('input_cols', v3.shape[1])

    # Split the data into training set and testing set 
    from sklearn.model_selection import train_test_split
    training_set, testing_set = train_test_split(to_train, test_size=0.2)

    # Set the columns for the variable and the target 
    x_train = training_set['var1(t-1)']
    y_train = training_set['var1(t)']
    x_test = testing_set['var1(t-1)']
    y_test = testing_set['var1(t)']
    
    # Reshape the data
    X_train = np.array(x_train).reshape(-1,1)
    Y_train = np.array(y_train).reshape(-1,1)
    X_test = np.array(x_test).reshape(-1,1)
    Y_test = np.array(y_test).reshape(-1,1)
    
    # Import the necessary library
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # create and fit the LSTM network
    deep_model = Sequential()
    deep_model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
    deep_model.add(Dense(1, kernel_initializer='normal'))
    deep_model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model on the training data
    deep_model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=2)

    # make predictions
    trainPredict = deep_model.predict(X_train)
    testPredict = deep_model.predict(X_test)

    # calculate root mean squared error
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    mlflow.log_metric("Train Score", trainScore)
    mlflow.sklearn.log_model(deep_model, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# In[ ]:


mlflow.pyfunc.save_model(
            path = './_model/test_model',
            python_model = NewModel(model)
        )


# In[27]:


import nbconvert


# In[29]:


get_ipython().system('jupyter nbconvert --to script deep_model.ipynb')


# In[ ]:




