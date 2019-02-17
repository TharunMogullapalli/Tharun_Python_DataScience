import pandas as pd

df_letters = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\letters.csv",sep=',')
df_letters.head()

Y = df_letters.iloc[:,0]
X= df_letters.iloc[:,1:16]

df_letters.groupby('lettr').count()

print(Y)
print(X)

Xnew = pd.get_dummies(X, sparse=False)
print(Xnew)

from sklearn.model_selection import train_test_split

Xtr,Xte,Ytr,Yte=train_test_split(Xnew,Y,random_state=0)
print(Xtr.shape,Xte.shape,Ytr.shape,Yte.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(Xtr,Ytr) #We first fit the training data of X, which contains the features. Then, we get the test dataset
Xtr_sc=scaler.transform(Xtr) #We then scale the data in the scaler variable
Xte_sc=scaler.transform(Xte)

from sklearn.svm import SVC
svc = SVC(random_state=2,kernel='rbf').fit(Xtr_sc,Ytr)
print('Test Accuracy of svc :%f'% svc.score(Xte_sc,Yte))

param_grid={'C':[0.01,0.1,1,10],'gamma':[0.001,0.01,0.05,0.07]} #These are the loop variables

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_search=GridSearchCV(SVC(),param_grid)  #Grid search is done with 2 loop variables
grid_search.fit(Xtr_sc,Ytr) #Model is getting fit
grid_search.score(Xte_sc,Yte)

print(grid_search.best_params_) #The parameters of grid search are C and gamma, which are its loop variables
print(grid_search.best_score_) #Gives the best score
