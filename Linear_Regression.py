import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#Prepare Data
from sklearn import datasets
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X.shape
diabetes_y.shape

#Split Data
from sklearn.model_selection import train_test_split
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.2, random_state = 0)

#Import LinearRegression and train algorithm
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(diabetes_X_train, diabetes_y_train)
print(reg.intercept_)
print(reg.coef_)

#Predicting
diabetes_y_pred = reg.predict(diabetes_X_test)
df = pd.DataFrame({'Actual': diabetes_y_test, 'Predicted': diabetes_y_pred})
df

#Evaluating the Algorithm 
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(diabetes_y_test, diabetes_y_pred))
print('MSE:', metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)))