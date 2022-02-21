# Pharmaceutical Sales Prediction

## Business Need
Rossman Pharmaceuticals has multiple stores across several cities and the finance team wants to forecast sales in all these stores across several cities six weeks ahead of time.
The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.
The objective here is to use the data provided to build and serve an end-to-end product that delivers the prediction to analysts in the finance team.

## Data and Features

### Data fields

<li>Id - an Id that represents a (Store, Date) duple within the test set</li>
<li>Store - a unique Id for each store</li>
<li>Sales - the turnover for any given day (this is what you are predicting)</li>
<li>Customers - the number of customers on a given day</li>
<li>Open - an indicator for whether the store was open: 0 = closed, 1 = open</li>
<li>StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None</li>
<li>SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools</li>
<li>StoreType - differentiates between 4 different store models: a, b, c, d</li>
<li>Assortment - describes an assortment level: a = basic, b = extra, c = extended.</li>
<li>CompetitionDistance - distance in meters to the nearest competitor store</li>
<li>CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened</li>
<li>Promo - indicates whether a store is running a promo on that day</li>
<li>Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating</li>
<li>Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2</li>
<li>PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store</li>

## Models
Different techniques were used in the project to train and serve the prediction, this was done to enable us to choose the best one. The techniques used are:
- Linear regression
- Random Forest
- Deep Learning (Long Shot-Term Memory - LSTM)

### Linear Regression
It is a linear approach to modelling and mapping the relationship between one variable, which is usually a target(y) and another variable or variables that are usually the features that determine the target. If regression analysis is known on a data that has just a single feature, it is said to be a univariate analysis, if it involves a dataset with multiple features, it is termed 'multivariate analysis'. The dataset used in this project has many features (as shown above) and so the analysis carried out is a multivariate analysis.

### Random Forests
Random forests or random decision forests, according to Wikipedia, are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For regression tasks, the mean or average prediction of the individual trees is returned. This algorithm was also used to make predictions. 

### LSTM
Long short-term memory is an artificial recurrent neural network architecture used in the field of deep learning. It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). LSTM networks are well suited for a classification task, processing and making predictions based on time series. Since LSTM is good for time series, we isolated the pharmaceutical data into time series data and created a deep learning model (LSTM) that is suitable for predictions. 

A streamlit app was created to serve the model. The app allows users to upload the data as a csv file and predict - the deep learning model is running under the hood. Here is the link https://github.com/SamDewriter/deployed_app to the repository that contains the code of the app.
