from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
breast_cancer.keys()

breast_cancer.data
breast_cancer.target
breast_cancer.feature_names
breast_cancer.target_names


import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(breast_cancer['data'], breast_cancer['target'], random_state=0, test_size=0.2)

print(Xtrain.shape, ytrain.shape)

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(Xtrain, ytrain)

print(knn)

predict = knn.predict(Xtest)

knn.score(Xtest, ytest)

knn.score(Xtrain, ytrain)

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
plt.ylabel('Accuracy Score') #k=27 is the best value

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,predict))
print(classification_report(ytest,predict,digits=4))
print(knn.score(Xtest,ytest))


