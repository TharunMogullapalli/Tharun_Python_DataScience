from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()

iris.DESCR

iris.feature_names

iris.target_names

iris.data[:5]

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(iris['data'], iris['target'], random_state=0, test_size=0.2)

print(Xtrain.shape, ytrain.shape)

knn = KNeighborsClassifier(n_neighbors=2)

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

Height = [151, 172, 137, 186, 129, 136, 179, 163, 152, 130]
Weight = [62, 80, 56, 92, 46, 57, 76, 72, 62, 49]


plt.scatter(Height,Weight)


from sklearn.linear_model import LinearRegression
simple_reg = LinearRegression().fit(pd.DataFrame(Height),pd.DataFrame(Weight))
simple_reg.score(pd.DataFrame(Height),pd.DataFrame(Weight))
simple_reg.predict([145,158,125])

simple_reg.coef_

simple_reg.intercept_

import statsmodels.api as sm
Weight = sm.add_constant(Weight)
simple_reg_model = sm.OLS(Weight,Height).fit()
simple_reg_model.summary()

simple_reg_model_y = sm.OLS(Height,Weight).fit()
simple_reg_model_y.summary()


import pandas as pd
data = pd.DataFrame(np.arange(16).reshape((4,4)),
                    index = ['ohio', 'Colorado', 'Utah', 'New York'],
                    columns= ['a','b','c','d'])
print(data)

data.drop('b',axis=1)

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc(ytest,knn.predict_proba(Xtest))
plt.show()

import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.1).fit(Xtrain,ytrain)
predict = mnb.predict(Xtest)
print("Test Accuracy Score is : %f" %mnb.score(Xtest,ytest))

print(mnb)

# generate confusion_matrix to check misclassification

from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,predict)
