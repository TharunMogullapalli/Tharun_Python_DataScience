import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns

df_adv = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\Advt data for multiple regression.csv", index_col = 0)

df_adv.info()

df_adv.head()

sns.pairplot(df_adv)

X = df_adv.iloc[:,0:3]
Y= df_adv.iloc[:,3]
from sklearn.model_selection import  train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(X,Y,random_state=0)
Xtr.shape; Yte.shape

print(Xtr.shape)
print(Yte.shape)
from sklearn.linear_model import LinearRegression
multi_reg1 = LinearRegression().fit(Xtr,Ytr)
multi_reg1.predict(Xte)
multi_reg1.score(Xtr,Ytr)
multi_reg1.score(Xte,Yte)

# residual analysis(' 4 in 1 1' plot)
residuals = Yte - multi_reg1.predict(Xte)

# plot residuals Vs predictor variables to detect trend/patterns

fig, axes = plt.subplots(2,2,figsize = (8,6))

axes[0,0].hist(residuals, color = 'maroon')
axes[0,0].set_title('Histogram of Residuals')

axes[0,1].scatter(Xte.iloc[:,0],residuals, color = 'blue')
axes[0,1].set_title('Predictor(Tv) vs Residuals')

axes[1,0].scatter(Xte.iloc[:,1],residuals, color = 'blue')
axes[1,0].set_title('Predictor(radio) vs Residuals')

axes[1,1].scatter(Xte.iloc[:,2],residuals, color = 'blue')
axes[1,1].set_title('Predictor(newspaper) vs Residuals')

plt.suptitle('Residual Plots of Multiple Linear Regression: Advt Dataset')

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
Xtrp = poly.fit_transform(Xtr)
Xtep = poly.fit_transform(Xte)

print(Xtrp)
print(Xtep)

from sklearn.linear_model import LinearRegression
multi_reg2 = LinearRegression().fit(Xtrp,Ytr)
multi_reg2.predict(Xtep)
multi_reg2.score(Xtrp,Ytr)
multi_reg2.score(Xtep,Yte)

df_bank = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\bank.csv", sep = ';')
df_bank.head()
df_bank.isna().sum()

df_banknew = df_bank.drop(['contact','day','month','poutcome'],axis=1)
df_banknew.shape
df_banknew[:2]
df_banknew.info()

import seaborn as sns
sns.pairplot(df_banknew, hue='education', size=1)

df_banknew.describe()
df_banknew.groupby(['marital','education'])['y'].count()

X = df_banknew.iloc[:,0:12]
X.shape

y = df_banknew.iloc[:,12]

Xnew = pd.get_dummies(X, drop_first= True)
Xnew.shape
df_banknew.groupby(['y']).y.count()

#Split Data Training and Test sets

from sklearn.model_selection import train_test_split
Xtrl, Xtel, Ytrl, Ytel = train_test_split(Xnew, y, random_state=0)
print(Xtrl.shape, Xtel.shape, Ytrl.shape, Ytel.shape)

# Feature Scaling (Normalizing Data)

from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler().fit(Xtrl) # Preferred when data are normally distributed else use MinMaxScaler

Xtr_sc = scaler.transform(Xtrl)
Xte_sc = scaler.transform(Xtel)

print(Xtr_sc)

Xtrl.head()
Ytrl.head()
from sklearn.linear_model import LogisticRegression

for C in [0.01,0.1,1,10,100] :
    logreg = LogisticRegression(C=C).fit(Xtr_sc, np.ravel(Ytrl,order='C'))

    pred_logreg = logreg.predict(Xte_sc)
    print('Accuracy is :%.6f'%logreg.score(Xte_sc,np.ravel(Ytel, order='C')),C)
    print('%.6f C='%logreg.score(Xte_sc,Ytel),C)

    logreg.score(Xtr_sc,Ytrl)

print(logreg)

import scikitplot as skplt
import matplotlib.pyplot as plt
skplt.metrics.plot_roc(Ytel, logreg.predict_proba(Xte_sc))
plt.show()



from sklearn.svm import SVC
svc = SVC(random_state=2, kernel=C).fit(Xtr_sc,Ytrl)
print('Test Accuracy of svc : %f'%svc.score(Xte_sc,Ytel))

for C in ['rbf','linear','poly'] :
       from sklearn.preprocessing import MinMaxScaler
       scaler = MinMaxScaler().fit(Xtrl)
       Xtr_sc01 = scaler.transform(Xtrl)
       Xte_sc01 = scaler.transform(Xtel)
       from sklearn.svm import SVC
       svc = SVC(random_state=2, kernel=C).fit(Xtr_sc01,Ytrl)
       print(svc)
       print('Test Accuracy is :%f'%svc.score(Xte_sc01,Ytel))

       pred_svc = svc.predict(Xte_sc01)
       from sklearn.metrics import confusion_matrix
       from sklearn.metrics import classification_report
       print(confusion_matrix(Ytel,pred_svc))
       print(classification_report(Ytel,pred_svc))

# To overcome the imbalanced data we use precision and recall

       from sklearn.metrics import confusion_matrix
       from sklearn.metrics import classification_report
       print(confusion_matrix(Ytel,pred_logreg))
       print(classification_report(Ytel,pred_logreg))


       from sklearn.svm import SVC
       svc = SVC(random_state=2, kernel=C, probability=True).fit(Xtr_sc,Ytrl)
       pred_svc = svc.predict(Xte_sc)


       import scikitplot as skplt
       import matplotlib.pyplot as plt
       skplt.metrics.plot_roc(Ytel, svc.predict_proba(Xte_sc))
       plt.title('ROC curves using SVC for bank data')

for C in ['rbf','linear','poly']:
       for P in [0.1,10,100]:
          for gamma in [0.01,10,100]:
           svc = SVC(random_state=2, kernel=C,C=P,gamma=gamma).fit(Xtr_sc01,Ytrl)
           print('Test Accuracy is: %f (C=)' %svc.score(Xte_sc01,Ytel),C)


