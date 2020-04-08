import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('cricket.csv')
dataset=dataset.drop(['PLAYER'],axis='columns')
dataset=dataset.replace(to_replace='-',value='0')

from sklearn.preprocessing import StandardScaler,normalize
scaler=StandardScaler()
dataset=scaler.fit_transform(dataset)
dataset=normalize(dataset)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x=pca.fit_transform(dataset)
x=pd.DataFrame(x)
x.columns=['X1','X2']


from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.0375,min_samples=4)
model.fit(x)
labels=model.labels_
print(np.unique(labels))

colors={}
colors[0]='r'
colors[1]='b'
colors[-1]='g'

cvec=[colors[label] for label in labels]
plt.figure(figsize=(10,7))
plt.title('DBSCAN')
plt.xlabel('X1')
plt.ylabel('X2')
r=plt.scatter(x['X1'],x['X2'],c='red',label='Cluster1')
b=plt.scatter(x['X1'],x['X2'],c='blue',label='Cluster2')
g=plt.scatter(x['X1'],x['X2'],c='green',label='Cluster3')
plt.scatter(x['X1'],x['X2'],c=cvec)
plt.legend()
plt.show()