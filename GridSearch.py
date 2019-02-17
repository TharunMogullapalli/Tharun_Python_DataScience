import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns

df_bank = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\bank.csv", sep=';')

banknew = df_bank.drop(['contact','day','month','poutcome'],axis=1)
X = banknew.iloc[:,0:12]
y= banknew.iloc[:,12]
Xnew = pd.get_dummies(X, sparse=False)
print(Xnew)
print(y)
from sklearn.model_selection import train_test_split

Xtr,Xte,Ytr,Yte=train_test_split(Xnew,y,random_state=0)
print(Xtr.shape,Xte.shape,Ytr.shape,Yte.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(Xtr,Ytr) #We first fit the training data of X, which contains the features. Then, we get the test dataset
Xtr_sc=scaler.transform(Xtr) #We then scale the data in the scaler variable
Xte_sc=scaler.transform(Xte)

from sklearn.svm import SVC
svc = SVC(random_state=2,kernel='rbf').fit(Xtr_sc,Ytr)
print('Test Accuracy of svc :%f'% svc.score(Xte_sc,Yte))

param_grid={'C':[0.01,0.1,1,10,100],'gamma':[0.001,0.01,0,1,10]} #These are the loop variables

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_search=GridSearchCV(SVC(),param_grid)  #Grid search is done with 2 loop variables
grid_search.fit(Xtr_sc,Ytr) #Model is getting fit
grid_search.score(Xte_sc,Yte)

print(grid_search.best_params_) #The parameters of grid search are C and gamma, which are its loop variables
print(grid_search.best_score_) #Gives the best score