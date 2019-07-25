#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[6]:


data = pd.read_csv(r'C:\Users\ony\Downloads\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\K_Means\Mall_Customers.csv')
print(data.head(n=10))
x = data.iloc[:, [3,4]].values


# In[25]:


wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=10,random_state =0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('no of cluster')
plt.ylabel('wcss')


# In[36]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++',random_state =0)
y = kmeans.fit_predict(x)


# In[37]:


plt.scatter(x[y == 0,0], x[y == 0,1], s = 100, c='red',label = '1')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 100, c='blue',label = '2')
plt.scatter(x[y == 2,0], x[y == 2,1], s = 100, c='green',label = '3')
plt.scatter(x[y == 3,0], x[y == 3,1], s = 100, c='cyan',label = '4')
plt.scatter(x[y == 4,0], x[y == 4,1], s = 100, c='magenta',label = '5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'centroids')
plt.title("Clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()


# In[ ]:




