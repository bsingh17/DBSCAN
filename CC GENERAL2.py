import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('CC GENERAL.csv')
dataset=dataset.drop(['CUST_ID'],axis='columns')
dataset.fillna(method='ffill',inplace=True)


from sklearn.preprocessing import StandardScaler,normalize
scaler=StandardScaler()
dataset=scaler.fit_transform(dataset)
dataset=normalize(dataset)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_pca=pca.fit_transform(dataset)
x_pca=pd.DataFrame(x_pca)
x_pca.columns=['P1','P2']

from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.0375,min_samples=50)
model.fit(x_pca)
labels=model.labels_
print(np.unique(labels))

colors={}
colors[0]='r'
colors[1]='g'
colors[-1]='k'
colors[2]='b'
colors[3]='c'
colors[4]='y'
colors[5]='m'

cvec=[colors[label] for label in labels]
r=plt.scatter(x_pca['P1'],x_pca['P2'],c='red',label='Cluster1')
g=plt.scatter(x_pca['P1'],x_pca['P2'],c='green',label='Cluster2') 
b=plt.scatter(x_pca['P1'],x_pca['P2'],c='black',label='Cluster3')
c=plt.scatter(x_pca['P1'],x_pca['P2'],c='cyan',label='Cluster4') 
y=plt.scatter(x_pca['P1'],x_pca['P2'],c='yellow',label='Cluster5')
m=plt.scatter(x_pca['P1'],x_pca['P2'],c='magenta',label='Cluster5')
k=plt.scatter(x_pca['P1'],x_pca['P2'],c='pink',label='Cluster6')
  
plt.figure(figsize =(9, 9)) 
plt.scatter(x_pca['P1'], x_pca['P2'], c = cvec) 
plt.show() 

