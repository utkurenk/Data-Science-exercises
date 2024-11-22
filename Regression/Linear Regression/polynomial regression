import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('Databases/Position_Salaries.csv')

x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

#linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x, y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(x_poly, y)

#visuals of linear regression
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visuals of the polynomial regression
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_reg_2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#overfitting model

#predict with linear regression
print(linear_reg.predict([[6.5]]))

#predict with polynomial regression
print(linear_reg_2.predict(poly_reg.fit_transform([[6.5]])))