# -*- coding: utf-8 -*-
"""Linear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a4WJL2-6BP7MjEJc9_W-BEiLWSRV0U51
"""

# Commented out IPython magic to ensure Python compatibility.
#Import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

#Prepare Data
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)

X=X[:, np.newaxis, 3]

X.shape

y.shape

#Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train

#Import LinearRegression and train algorithm
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.intercept_)
print(reg.coef_)

#Predicting
y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

"""# New Section"""

#Evaluating the Algorithm 
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(X_test[:,0], y_test, color="black")
plt.plot(X_test[:,0], y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()