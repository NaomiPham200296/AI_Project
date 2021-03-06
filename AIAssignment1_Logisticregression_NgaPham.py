# -*- coding: utf-8 -*-
"""AIAssignment1_LogisticRegression_NgaPham.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GPLBUi4hPtLV9pu3W_1Vkg0-_A8ktKdK
"""

# Commented out IPython magic to ensure Python compatibility.
#Import Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

#Import data
from google.colab import files
uploaded = files.upload()

data = pd.read_csv('voice.csv')

data.head()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

#correlation matrix
df = pd.DataFrame(data)
corrMatrix = df.corr()
corrMatrix

#Heat map
import seaborn as sns
sns.heatmap(corrMatrix)

#New dataset
data = data.drop('median', 1)
data = data.drop('Q75', 1)
data = data.drop('sfm', 1)
data = data.drop('dfrange', 1)

data

#Train algorithm
classifier=LogisticRegression()
classifier.fit(X_train_new, y_train_new)

y_pred_new=classifier.predict(X_test_new)

#measure the accuracy on new test set
accuracy_score(y_test_new,y_pred_new)

"""# New Section"""