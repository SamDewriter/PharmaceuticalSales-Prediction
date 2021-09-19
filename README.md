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
