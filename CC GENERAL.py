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
x_pca.columns=['X1','X2']

from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.0375,min_samples=3)
model.fit(x_pca)
labels=model.labels_


colors={}
colors[0]='y'
colors[1]='g'
colors[2]='r'
colors[-1]='m'

cvec=[colors[label] for label in labels]
plt.figure(figsize=(10,10))
plt.title('Cluster Representation')
plt.xlabel('X1')
plt.ylabel('X2')
y=plt.scatter(x_pca['X1'],x_pca['X2'],c='yellow',label='Cluster1')
g=plt.scatter(x_pca['X1'],x_pca['X2'],c='green',label='Cluster2')
r=plt.scatter(x_pca['X1'],x_pca['X2'],c='red',label='Cluster3')
m=plt.scatter(x_pca['X1'],x_pca['X2'],c='magenta',label='Cluster4')
plt.scatter(x_pca['X1'],x_pca['X2'],c=cvec)
plt.legend()
plt.show()