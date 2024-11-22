import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('Databases/50_Startups.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#dummy variable trap deleting 1 column to get rid of it
x = np.delete(x, 0, 1)

#train - test sets split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#no need to apply feature scaling in multiple linear scaling

#regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#prediction
y_prediction = regressor.predict(x_test)
np.set_printoptions(precision=2)
#changing row to column 
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), y_test.reshape(len(y_test), 1)), 1))

regressor.coef_
regressor.intercept_

print(regressor.score(x_train, y_train))

plt.scatter(y_test, y_prediction)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_prediction)', size=18)
plt.show()