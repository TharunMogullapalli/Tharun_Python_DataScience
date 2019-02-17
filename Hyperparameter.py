from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# Load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Construct some pipelines
pipe_knn = Pipeline([('scl', StandardScaler()),
                    ('clf', KNeighborsClassifier())])

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42))])

pipe_df = Pipeline([('scl', StandardScaler()),
                        ('clf', DecisionTreeClassifier(random_state=42))])

pipe_rf = Pipeline([('scl', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))])

pipe_nb = Pipeline([('scl', MinMaxScaler()),
                    ('clf', MultinomialNB())])

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', svm.SVC(random_state=42))])

pipe_mlp = Pipeline([('scl', MinMaxScaler()),
                    ('clf', MLPClassifier(random_state=42))])

# Set grid search params
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1]
param_range_m1 =  [100,200,300,400]
param_range_m2 = (100,2)

grid_params_knn = [{'clf__n_neighbors': param_range}]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': param_range_fl
                 }]

grid_params_df = [{'clf__criterion': ['gini', 'entropy']}]

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__n_estimators': param_range,
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:]}]

grid_params_nb = [{'clf__alpha': param_range_fl}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf', 'poly'],
                    'clf__C': param_range_fl}]

grid_params_mlp = [{'clf__max_iter': param_range_m1,
                   'clf__hidden_layer_sizes': param_range_m2
                   }]

# Construct grid searches
jobs = -1

gs_knn = GridSearchCV(estimator=pipe_knn,
                     param_grid=grid_params_knn,
                     scoring='accuracy',
                     cv=10)

gs_lr = GridSearchCV(estimator=pipe_lr,
                     param_grid=grid_params_lr,
                     scoring='accuracy',
                     cv=10)

gs_df = GridSearchCV(estimator=pipe_df,
                     param_grid=grid_params_df,
                     scoring='accuracy',
                     cv=10)


gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=grid_params_rf,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=jobs)

gs_nb = GridSearchCV(estimator=pipe_nb,
                     param_grid=grid_params_nb,
                     scoring='accuracy',
                     cv=10)

gs_svm = GridSearchCV(estimator=pipe_svm,
                      param_grid=grid_params_svm,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=jobs)

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                      param_grid=grid_params_mlp,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=jobs)



# List of pipelines for ease of iteration
grids = [gs_knn,gs_lr,gs_df,gs_rf,gs_nb,gs_svm,gs_mlp]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'KNeighborsClassifier', 1: 'Logistic Regression',
             2: 'DecisionTree', 3: 'Random Forest',
             4: 'Naive Byes',5: 'Support Vector Machine',6: 'Neural Networks'}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])