#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[21]:


data = pd.read_csv(r'C:\Users\ony\Downloads\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 16 - Support Vector Machine (SVM)\SVM\Social_Network_Ads.csv')


# In[6]:


print(data.head(n=10))
print(data.isnull().values.sum())


# In[25]:


x = data.iloc[:,[2,3]].values
y = data.iloc[:, 4:].values


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
Sc_x = StandardScaler()
x_train = Sc_x.fit_transform(x_train) 
x_test = Sc_x.fit_transform(x_test) 


# In[10]:


from sklearn.svm import SVC
cls = SVC(kernel= 'linear',random_state = 0)
cls.fit(x_train,y_train)
pred = cls.predict(x_test)
print(cls.score(x_test,y_test))


# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
print(cm)


# In[28]:


from matplotlib.colors import ListedColormap 
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01), np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+  1, step = 0.01))
plt.contourf(X1,X2, cls.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(
                 ('Yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
               c = ListedColormap(('Red',"green"))(i),label = j)
plt.title("SVM(training set)")
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()


# In[27]:


from matplotlib.colors import ListedColormap 
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01), np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+  1, step = 0.01))
plt.contourf(X1,X2, cls.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(
                 ('Yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
               c = ListedColormap(('Red',"green"))(i),label = j)
plt.title("SVM(test set)")
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()


# In[ ]:




