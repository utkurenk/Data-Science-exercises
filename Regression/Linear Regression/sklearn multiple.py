import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

data = pd.read_csv('real_estate_price_size_year.csv')
data.describe()

x = data[['size', 'year']]
y = data['price']

reg = LinearRegression()
reg.fit(x, y)

reg.coef_
reg.intercept_

r2 = reg.score(x, y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print(adjusted_r2)

from sklearn.feature_selection import f_regression

f_regression(x, y)
p_values = f_regression(x,y)[1]
p_values.round(3)

reg_summary = pd.DataFrame(data = x.columns.values, columns= ['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p values'] = p_values.round(3)
print(reg_summary)
