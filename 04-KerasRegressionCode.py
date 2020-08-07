""" Here one focuse more on feature engineering with realistic data set

I will be using data from a Kaggle data can be found:

https://www.kaggle.com/harlfoxem/housesalesprediction

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option('display.max_columns',50)


house = pd.read_csv('kc_house_data.csv')
# first let us see if we have any missing data in each columns
# Looking for nulls
house.isnull().any()
house.isnull().sum()
# let us see the features name
house.columns

house.describe().transpose()

#%%
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                 hue='bedrooms', palette='tab20',size=1.4)
g.set(xticklabels=[]);


plt.subplots(figsize=(17,14))
sns.heatmap(house.corr(),annot=True,linewidths=0.5,linecolor="Black",fmt="1.1f")
plt.title("Data Correlation",fontsize=50)
plt.show()

#%%
# in order to have a better idea let us plot the price and see its distribution
plt.figure(figsize=(10,5))
sns.distplot(house['price'])

# as we can see the are some extream points (expensive houses which are not that many)
# so it is better to skip them when we build the model

sns.pairplot(house[['sqft_above','price','sqft_living','bedrooms']],hue='bedrooms', palette='husl');

sns.countplot(x= house['bedrooms'])
# we can see it is extended u tp 33 but it is a small bar and not visible
sns.pairplot(house, hue='price')
# now let us see the correlation of our target with other value
house.corr()['price'].sort_values()
# we see "sqft_living" has a strong correlation with price

sns.scatterplot(x= house['sqft_living'], y = house['price'])
sns.scatterplot(x=house['grade'], y = house['price'])
sns.scatterplot(x=house['bedrooms'], y = house['price'])

# in our data we have "lattitude" and "longitude" which give the house position in
# king country, USA
plt.figure(figsize=(5,8))
sns.scatterplot(x= house['long'],y =house['lat'], hue = house['price'])

# because of those extream points we are not getting a good
# color distribution. so now let us drop those extreame points

house2 = house.sort_values('price',ascending=False)
len(house)*(0.01)  #216
non_top_1_perc = house2[216:]

#non_top_1_perc = house.sort_values('price',ascending=False).iloc[216:]

# now let us plot it again, to get more color distribution

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)

#%%% 138
"""Feature engineering """
# now we can do some feature engineerin from data
# let us drop some featuers which is not very informatic in given data set
# we look at data again
house.head(5)
house = house.drop('id', axis=1)

house['date']
# next if we look at the date it shows some sort of the string "" dtype: object"".
# So we need to convert it by doing the following, we can use "to_datetime" function from Pandas

house['date'] = pd.to_datetime(house['date'])
# now the type of date is ""dtype: datetime64[ns]""

# this is called feature engineering, because these features  are hidden inside of
# string date. now we try to exteact or engineering more information off the original data

house['year'] = house['date'].apply(lambda date:date.year)
house['month'] = house['date'].apply(lambda date:date.month)

# we can do some explatory visualization to see the impact of month on house selling

sns.boxplot(x='year',y='price',data=house)

house.groupby('month').mean()['price'].plot()

house = house.drop('date', axis = 1)

# now if we look at the zipcode column we see we have 70 uniqu values
# which is basically too much to catagorize the data
house['zipcode'].value_counts()

# So I go ahead and for this particular example I drop this colomn

house = house.drop('zipcode',axis=1)
# could make sense due to scaling, higher should correlate to more value

# other thing er can d is looking at continoues or categorical data
house['yr_renovated'].value_counts()
house['sqft_basement'].value_counts()

#%%139
"""Scaling and Train Test Split"""

X = house.drop('price',axis=1).values
y = house['price'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# we can save time by fitting ans transfering at the same time
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape


""" Creating a Model"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


model = Sequential()
# typcally what we do is we try to base the number of neurons
# in our layer from the size of actual feature data
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

"""Training the Model"""
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=400)


#140
losses = pd.DataFrame(model.history.history)
# this data frame has two columns, one is loss and the other is called
# val_loss ---> this is loss on that test set
# that validation data and now I can directly compare the loss on training
# and loss on test data in order to see if i am overfitting to the training
# data on my model. Simply we can plot it
losses.plot()

#%%% we can do some evaluationon our test data 140
"""Evaluation on Test Data"""
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

predictions = model.predict(X_test)

mean_absolute_error(y_test,predictions)
house['price'].mean()

explained_variance_score(y_test,predictions)

# Our predictions
plt.scatter(y_test,predictions)
# Perfect predictions
plt.plot(y_test,y_test,'r')


















