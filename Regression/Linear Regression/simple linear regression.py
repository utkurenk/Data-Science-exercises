import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sns.set()

data = pd.read_csv('real_estate_price_size.csv')
data.describe()

y = data['price']
x1 = data['size']

plt.scatter(x1, y)
yhat = 223.1787 * x1 + 101900
fig = plt.plot(x1, yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel('size', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
