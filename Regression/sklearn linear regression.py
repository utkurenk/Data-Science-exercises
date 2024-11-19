import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

data = pd.read_csv('real_estate_price_size.csv')

x = data['size']
y = data['price']

x_matrix = x.values.reshape(-1,1)

reg = LinearRegression()
reg.fit(x_matrix, y)

print(reg.score(x_matrix, y)) #R squared
reg.coef_
reg.intercept_

print(reg.predict([[500]]))