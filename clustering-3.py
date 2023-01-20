# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:51:25 2023

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

plt.style.use('Solarize_Light2')
df = pd.read_excel('C:/Users/HP/Downloads/Electricity.xls')
#datatranspose= df.set_index('Country Name').transpose()

df = df[113:261]
print(df)
   
df.shape
df.describe()

X1 = df.loc[:, ['2010','2020']].values
inertia = []
for n in range(1 , 20):
    model = KMeans(n_clusters = n,
               init='k-means++',
               max_iter=500,
               random_state=42)
    model.fit(X1)
    inertia.append(model.inertia_)
    plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 20) , inertia , 'o')
plt.plot(np.arange(1 , 20) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

#We can see that the "elbow" on the graph above (where the interia becomes more linear) is at K=2. We can then fit our K-means algorithm one more time and plot the different clusters assigned to the data:
model = KMeans(n_clusters = 3,
            init='k-means++',
            max_iter=500,
            random_state=42)
model.fit(X1)
labels = model.labels_
centroids = model.cluster_centers_
y_kmeans = model.fit_predict(X1) 

plt.figure(figsize=(20,10))
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('2010')
plt.ylabel('2020')
plt.legend()
plt.show()
     

     