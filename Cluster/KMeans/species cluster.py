import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Databases/iris_dataset.csv')

#plt.scatter(data['sepal_length'], data['sepal_width'])
#plt.xlim(4,8)
#plt.ylim(1,5)

x = data.copy()

kmeans = KMeans(2)
kmeans.fit(x)

#unscaled
identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters

#plt.scatter(data['sepal_length'], data['sepal_width'], c=data_with_clusters['Cluster'], cmap='rainbow')
#plt.xlim(4,8)
#plt.ylim(1,5)
#plt.show()

#scaled
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)

kmeans_scaled = KMeans(2)
kmeans_scaled.fit(x)
scaled_clusters = kmeans_scaled.fit_predict(x_scaled)

data_scaled_clusters = data.copy()
data_scaled_clusters['Cluster'] = scaled_clusters

plt.scatter(data['sepal_length'], data['sepal_width'], c=data_scaled_clusters['Cluster'], cmap='rainbow')
plt.xlim(4,8)
plt.ylim(1,5)
plt.show()

#wcss
wcss = []
for i in range(1,20):
    kmeans_function = KMeans(i)
    kmeans_function.fit(x_scaled)
    wcss_iter = kmeans_function.inertia_
    wcss.append(wcss_iter)

print(wcss)