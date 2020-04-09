import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('crop1.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

labelencoder_X1 = LabelEncoder()
X[:, 3] = labelencoder_X1.fit_transform(X[:, 3])

onehotencoder1 = OneHotEncoder(categorical_features = [2,3])
X = onehotencoder1.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =30, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

import math
for j in range(len(y_pred)):
    y_pred[j] = math.ceil(y_pred[j])

list_crops = {}
Y = dataset.iloc[:,0].values
y_pred = np.array(y_pred,dtype=int)
crops = []

for i in range(len(y_pred)):
    list_crops = dataset[dataset['Cost'] > y_pred[i]]
    
list_crops = list_crops[list_crops['Cost']>2000]
K = input()
if K == 'Rabi':
    list_crops = list_crops[list_crops['Season']!='Kharif']
    #list_crops = list_crops.iloc[:,0].values
if K == 'Kharif':
    list_crops = list_crops[list_crops['Season']!='Rabi']
    #list_crops = list_crops.iloc[:,0].values
print(list_crops.iloc[:,0].values)

    
