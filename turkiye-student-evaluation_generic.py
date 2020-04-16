import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('turkiye-student-evaluation_generic.csv')

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(dataset)
x=scaler.fit_transform(dataset)



from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x_pca=pca.transform(x)
x_pca=pd.DataFrame(x_pca)
x_pca.columns=['X1','X2']

from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.5,min_samples=10)
model.fit(x_pca)
labels=model.labels_
print(np.unique(labels))

colors={}
colors[0]='b'
colors[1]='r'
colors[-1]='g'

cvec=[colors[label] for label in labels]
plt.figure(figsize=(10,7))
plt.title('DBSCAN')
plt.xlabel('X1')
plt.ylabel('X2')
b=plt.scatter(x_pca['X1'],x_pca['X2'],c='blue',label='Cluster1')
r=plt.scatter(x_pca['X1'],x_pca['X2'],c='red',label='Cluster2')
g=plt.scatter(x_pca['X1'],x_pca['X2'],c='green',label='Clsuter3')
plt.scatter(x_pca['X1'],x_pca['X2'],c=cvec)
plt.legend()
plt.show()

