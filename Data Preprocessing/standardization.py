import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

data = pd.read_csv('Databases/real_estate_price_size_year.csv')
data.describe()

x = data[['size', 'year']]
y = data['price']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y)
reg.coef_
reg.intercept_

reg_summary = pd.DataFrame([['Bias'],['size'], ['year']], columns= ['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print(reg_summary)
