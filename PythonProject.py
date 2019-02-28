import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Import the Data-set
Passengers = pd.read_csv("C:\\Users\\Srivas\\Downloads\\Air_Traffic_Passenger_Statistics.csv",sep=',')
Passengers.keys()
Passengers.info()
Passengers.isna().sum()

#Drop the Unnecessary features
Passengers_New = Passengers.drop(columns = ['Activity Period', 'Operating Airline IATA Code','Published Airline IATA Code','Terminal', 'Published Airline'])
Passengers_New.info()

#Exploratory Data Analysis
print(Passengers_New['Passenger Count'].describe())
sns.distplot(Passengers_New['Passenger Count'], color='g', bins=50)#Data is not normal to overcome this we transform the data

PriceCategoryCount = Passengers_New.groupby(['Price Category Code']).count()
print(PriceCategoryCount)#Data is imbalanced to overcome this we use precision and recall

sns.catplot(x='Passenger Count', y ='GEO Region',hue='Price Category Code',data=Passengers_New)

#Features
X=Passengers_New.iloc[:,0:6]

#Price Category Code--Binary Label
Y=Passengers_New.iloc[:,6]

#Label Encoding
label_encoder = LabelEncoder()

YNew = label_encoder.fit_transform(Y)
print(YNew.shape)

XNew = X.apply(label_encoder.fit_transform)
print(XNew.shape)

#Training set & Test set split
XNew_train, XNew_test, YNew_train, YNew_test = train_test_split(XNew, YNew, test_size=0.4, random_state=42)
print(XNew_train.shape,XNew_test.shape,YNew_train.shape,YNew_test.shape)

#Construct classification pipelines
pipe_knn = Pipeline([('scl', StandardScaler()),
                    ('clf', KNeighborsClassifier())])

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', svm.SVC(random_state=42))])

#Set grid search params
param_range = range(1,25)
param_range_fl = [1.0, 0.5, 0.7]

grid_params_knn = [{'clf__n_neighbors': param_range}]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': param_range_fl
                 }]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf', 'poly'],
                    'clf__C': param_range_fl,
                    'clf__gamma': param_range_fl}]


# Construct grid searches
jobs = 1

gs_knn = GridSearchCV(estimator=pipe_knn,
                      param_grid=grid_params_knn,
                      scoring='accuracy',
                      cv=10)

gs_lr = GridSearchCV(estimator=pipe_lr,
                     param_grid=grid_params_lr,
                     scoring='accuracy',
                     cv=10)

gs_svm = GridSearchCV(estimator=pipe_svm,
                      param_grid=grid_params_svm,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=jobs)

# List of pipelines for ease of iteration
grids = [gs_knn,gs_lr,gs_svm]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'KNeighborsClassifier', 1: 'Logistic Regression',
             2: 'Support Vector Machine'}

# Logging for Visual Comparison
log_cols = ["Classifier", "Training Score", "Test Score"]
log = pd.DataFrame(columns=log_cols)

# Fit the grid search objects
print('Performing multiple classification models...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    name = grid_dict[idx]
    # Fit grid search
    gs.fit(XNew_train, YNew_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(XNew_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(YNew_test, y_pred))
    print(confusion_matrix(YNew_test, y_pred))
    print(classification_report(YNew_test, y_pred))

    log_entry = pd.DataFrame([[name,gs.best_score_,accuracy_score(YNew_test, y_pred)]], columns=log_cols)
    log = log.append(log_entry)

    # Track best (highest test accuracy) model
    if accuracy_score(YNew_test, y_pred) > best_acc:
        best_acc = accuracy_score(YNew_test, y_pred)
        best_gs = gs
        best_clf = idx

print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

#shows the results for all the classifiers
print(log)

#Plotting the Test & Training Accuracy Scores
sns.scatterplot(x='Training Score', y='Test Score',hue='Classifier', style='Classifier', data=log)
plt.xlabel('Training Score')
plt.ylabel('Test Score')
plt.title('Accuracy Scores')
plt.show()


