import pandas as pd

df_cars = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\Car_Evaluations.csv",sep=',')
df_cars.head()

df_cars.groupby('CLASS').CLASS.count()

X = df_cars.iloc[:,0:6]
Y = df_cars.iloc[:,6]

X.head()
Y.head()

Xnew = pd.get_dummies(X,drop_first=False)
Xnew.head()
Ynew = pd.get_dummies(Y, drop_first=False)
Ynew.head()

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(Xnew,Ynew, random_state= 0)
print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)
from sklearn.tree import DecisionTreeClassifier
for C in [5,6,7]:
      dtree = DecisionTreeClassifier(max_depth=C, random_state=0,criterion='entropy').fit(Xtr,ytr)
      predict = dtree.predict(Xte)
      print('Training Set Accuracy:%.4f' %dtree.score(Xtr,ytr))
      print('Test set Accuracy:%.4f' %dtree.score(Xte,yte))

from sklearn.metrics import confusion_matrix
confusion_matrix(yte.values.argmax(axis=1),predict.argmax(axis=1))

# Random Forest

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=33 , random_state= 42,criterion='entropy')
forest.fit(Xtr,ytr)
qredict = forest.predict(Xte)

print('Training Set Accuracy:%.4f' %forest.score(Xtr,ytr))
print('Test set Accuracy:%.4f' %forest.score(Xte,yte))

from sklearn.metrics import  confusion_matrix
confusion_matrix(yte.values.argmax(axis=1), qredict.argmax(axis=1))
print(forest)
