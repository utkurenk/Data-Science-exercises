import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Countries-exercise.csv')
print(data.head())

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-100, 100)
plt.ylim(-90,90)
#plt.show()

#slice data frame
x = data.iloc[:,1:3]
print(x)

#cluster
kmeans = KMeans(5)
kmeans.fit(x)

#results
identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_cluster
print(data_with_clusters)

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-200, 200)
plt.ylim(-200,200)
plt.show()