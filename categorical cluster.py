import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Categorical.csv')
print(data)

#categorical
data_mapped = data.copy()
data_mapped['continent'] = data_mapped['continent'].map({'North America':0,'Europe':1,'Asia':2,'Africa':3,'South America':4, 'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})
print(data_mapped)

x = data_mapped.iloc[:,1:4]

#cluster
kmeans = KMeans(7)
kmeans.fit(x)

#results
identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_cluster
print(data_with_clusters)

#plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
#plt.xlim(-200, 200)
#plt.ylim(-200,200)

#WCSS
wcss = []
for i in range(1,20):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)

#elbow method
number_clusters = range(1,20)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()