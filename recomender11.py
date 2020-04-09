import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('crop1.csv')
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X2 = LabelEncoder()
y = labelencoder_X2.fit_transform(y)

labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

labelencoder_X1 = LabelEncoder()
X[:, 3] = labelencoder_X1.fit_transform(X[:, 3])

onehotencoder1 = OneHotEncoder(categorical_features = [2,3])
X = onehotencoder1.fit_transform(X).toarray()
X = X[:,1:]
# Feature Scaling



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =35, random_state = 0)
regressor.fit(X, y)
# Predicting a new result
y_pred = regressor.predict(X_test)

import statsmodels.regression.linear_model as sm

X = np.append(arr = np.ones((46,1)).astype(int),values = X, axis=1)

X_opt = X
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,[0,1,2,3,4,5,6,7,9]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,[0,1,2,3,5,6,7,8]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,1:]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,[0,1,2,3,4,6]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X_opt[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,[0,1,2,4]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:,[0,2,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =32, random_state = 0)
regressor.fit(X_opt, y)
# Predicting a new result
y_pred = regressor.predict(X_test)

import math
for j in range(len(y_pred)):
    y_pred[i] = math.ceil(y_pred[i])

list_crops = []
Y = dataset.iloc[:,0].values
y_pred = np.array(y_pred,dtype=int)
for i in range(len(y_pred)):
    k = Y[y_pred[i]]
    list_crops.append(k)
    
print(list_crops)
