#!/usr/bin/env python
# coding: utf-8

# In[1]:
import dvc.api

# data analysis and wrangling
import pandas as pd
import numpy as np

path = 'dvc-data/to_train.csv' 
repo = 'C:/Users/daud/Desktop/Twitter/Week_3'
version = 'v1'

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version,
    )

mlflow.set_experiment('Modelling')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # read the data
    to_train = pd.read_csv(data_url)
    
    # log parameters
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', to_train.shape[0])
    mlflow.log_param('input_cols', to_train.shape[1])

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
    







