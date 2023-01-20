# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:51:25 2023

@author: HP
"""
#importing librarys
from scipy.optimize import curve_fit
from numpy import arange
from matplotlib import pyplot
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
#importing sklearn library for machine learning algorithms
import sklearn.metrics as skmet
import matplotlib.pyplot as plt

# import the csv file which contain the dataset
dataset="C:/Users/HP/Documents/World bank.csv "
text_file = open(dataset,'r')
line_list = text_file.readlines()

with open("output.csv", "w") as txt_file:
    for line in line_list[4:]:
        txt_file.write(line)

df = pd.read_csv("output.csv",engine="python")
df=df.fillna(0)
df=df.drop(["Unnamed: 66"],axis=1)
indicators=["SP.URB.TOTL","AG.LND.FRST.K2"]
res=df[df['Indicator Code'].isin(indicators)]
columns=res.columns
temp_df=res.drop(columns[:4],axis=1).transpose()
res["output"]=temp_df.mean()

res=res[["Country Name","Indicator Code","output"]]
urban_pop=res[df["Indicator Code"]=="SP.URB.TOTL"]
forest_area=res[df["Indicator Code"]=="AG.LND.FRST.K2"]
countryCode=list(set(list(res["Country Name"])))
data=np.empty((0, 3))
for i in countryCode:
  if(forest_area[forest_area['Country Name']==i]["output"].empty):
    continue
  forestArea=float(forest_area[forest_area['Country Name']==i]["output"])
  UrbanPopulation=float(urban_pop[urban_pop['Country Name']==i]["output"])
  data=np.append(data,np.array([[i,forestArea,UrbanPopulation]]),axis=0)
data
# initializing clustering the data
ncluster = 4
xy=data[:,1:]
x=data[:,1]
y=data[:,2]

# assigning each data point to a cluster. 
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(xy)
labels = kmeans.labels_
centroid = kmeans.cluster_centers_
print(centroid)
# calculate the number of clusters
print(skmet.silhouette_score(xy, labels))
# to pick a colour give labels
plt.figure(figsize=(10, 10))
column = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(0,ncluster): 
  plt.plot(x[labels==l], y[labels==l], "o", markersize=4, color=column[l])
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
# show cluster centres

country=data[:,0]
for i in range(ncluster):
  print("Cluster="+str(i))
  print(country[labels==i])

result_df=df[df['Indicator Code'].isin(indicators)]
result_df=result_df[result_df["Country Name"]=="Canada"]
result_df=result_df.transpose()
temp_df=result_df.drop(columns[:4],axis=0)
temp_df=temp_df.drop(columns[-10:],axis=0)
temp_df.columns=["x","y"]
y=list(temp_df["y"])
x=[ int(x) for x in temp_df.index]

#initializing chi-square fitting

def objective(x, a, b):
  return a*x+b

popt,covar= curve_fit(objective, x, y,absolute_sigma=True)
# summarize the parameter values
print(popt)
a, b = popt
sigma = np.sqrt(np.diag(covar))
print('y = %.5f * x+%.5f' % (a, b))
x_line = arange(min(x), 2027, 1)
y_line=list()
for i in x_line:
  y_line.append(objective(i, a, b))
labels=list()
pyplot.plot(x, y,label="data")
pyplot.plot(x_line, y_line,label="best fit")
plt.xlabel(r'year', fontsize=20)
plt.ylabel(r'Forest area', fontsize=20)
plt.legend(bbox_to_anchor =(1.75, 1.15), ncol = 5)
f = plt.figure()
f.set_figwidth(1000)
f.set_figheight(1000)
plt.show()

def err_ranges(x, func, param, sigma):
    import itertools as iter
    
    # start arrays for the lower and upper limits 
    lower = [func(i, *param) for i in x]
    upper = lower
    
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = [func(i, *p) for i in x]
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    print(lower)
    print(upper)  
    return lower, upper 
lower,upper=err_ranges(x,objective,[a,b],sigma)

forecast = [objective(i,a,b) for i in x]
plt.figure()
plt.plot(x, y, label="Forest area")
plt.plot(x, forecast, label="predict")
plt.fill_between(x, lower, upper, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("sq.km")
plt.legend()
plt.show()
     

     