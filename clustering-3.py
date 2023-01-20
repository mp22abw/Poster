# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:51:25 2023

@author: HP
"""
#importing librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing seaborn library to visualize random distributions.
import seaborn as sns
#importing plotly to style figures
import plotly.express as px
#importing sklearn library for machine learning algorithms
from sklearn.cluster import KMeans

plt.style.use('Solarize_Light2')
#reading data from the dataset
df = pd.read_excel('C:/Users/HP/Downloads/Electricity.xls')

#selecting data

df = df[113:260]
print(df)
   
df.shape
df.describe()

'''To find best value of K,we need to run K-means across our data for a range of
 possible values.We only have 812 data points, so the maximum number of clusters is 812.'''

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

'''We can see that the "elbow" on the graph above (where the interia becomes more linear)
 is at K=3'''
# Train with k means clustering

Km = KMeans(n_clusters = 3,
            init='k-means++',
            max_iter=500,
            random_state=42)
Km.fit(X1)
labels = Km.labels_
centroids = Km.cluster_centers_
# assigning each data point to a cluster.
y_kmeans = model.fit_predict(X1) 

#plotting the scatter plot by appyling the cluster algorithm  

plt.figure(figsize=(20,10))
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('2010')
plt.ylabel('2020')
plt.legend()
plt.show()
     

     