import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_wine = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\wines.csv",sep=',')
df_wine.info()
df_wine.head()

X= df_wine.iloc[:,1:14]
Y = df_wine.iloc[:,0]


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y,random_state=0)
print(Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain_sc = scaler.transform(Xtrain)
Xtest_sc = scaler.transform(Xtest)
print(Xtrain_sc)
print(ytrain)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500).fit(Xtrain_sc,ytrain)
predict_mlp = mlp.predict(Xtest_sc)

print('Training Set accuracy score:%.4f'%mlp.score(Xtrain_sc,ytrain))
print('Training Set accuracy score:%.4f'%mlp.score(Xtest_sc,ytest))
print(confusion_matrix(ytest,predict_mlp))


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca_projected = pca.fit_transform(X)
print(pca_projected)

plt.scatter(pca_projected[:,0],pca_projected[:,1],cmap='jet',c=Y)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


#Use KNN Classification on the new dataset
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(pca_projected,Y,random_state=0,test_size=0.2)
print(Xtrain.shape,ytrain.shape)


from sklearn.neighbors import KNeighborsClassifier

scoretest_i=[]
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain,ytrain)
    predtest_i = knn.predict(Xtest)
    scoretest_i.append(knn.score(Xtest,ytest))

scoretrain_i=[]
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain,ytrain)
    predtrain_i = knn.predict(Xtrain)
    scoretrain_i.append(knn.score(Xtrain,ytrain))


#Plot the k values
plt.figure(1)
plt.plot(np.arange(1,40), scoretest_i, color = 'red', linestyle = 'dashed',
         marker = 'o', markerfacecolor = 'blue', markersize = 12)
plt.plot(np.arange(1,40), scoretrain_i, color = 'green', linestyle = 'dashed',
         marker = 'o', markerfacecolor = 'pink', markersize = 12)
plt.title('Accuracy Scores for k values 1- 40')
plt.xlabel('K value')
plt.ylabel('Accuracy Score')


pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker = 'o', markersize = 10, markeredgecolor = 'red', markerfacecolor = 'blue')

plt.xlabel('number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Variance Vs No of components')