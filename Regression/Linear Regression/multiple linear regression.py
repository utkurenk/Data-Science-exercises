import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sns.set()

data = pd.read_csv('real_estate_price_size_year.csv')
data.describe()

y = data['price']
x1 = data[['size', 'year']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())