import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
iris.keys()
label_names = iris['target_names']
labels = iris['target']
feature_names = iris['feature_names']
features = iris['data']


classifiers = [
    KNeighborsClassifier(n_neighbors=4),
    SVC(kernel="poly", C=1.2),
    DecisionTreeClassifier(criterion='entropy',max_depth=100),
    RandomForestClassifier(n_estimators=60, criterion='entropy'),
    LogisticRegression(C=0.6,multi_class='auto'),
    MultinomialNB(alpha=0.6),
    MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)]

# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.3,
                                                          random_state=42)

#features scaling
scaler = MinMaxScaler()
scaler.fit(train_features)
Xtrain_sc = scaler.transform(train_features)
Xtest_sc = scaler.transform(test_features)


# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(Xtrain_sc, train_labels)
    name = clf.__class__.__name__

    print("=" * 50)
    print(name)

    print('****Results****')
    train_predictions = clf.predict(Xtest_sc)
    acc = accuracy_score(test_labels, train_predictions)
    print("Accuracy: {:.4%}".format(acc))


    log_entry = pd.DataFrame([[name, acc * 100]], columns=log_cols)
    log = log.append(log_entry)

print("=" * 50)


sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

####K-Means

X = iris.data
y = iris.target

from sklearn.cluster import KMeans
km = KMeans(n_clusters= 8)
km.fit(X)
km.predict(X)

#plot the model

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2],c=km.labels_,edgecolor = "k",s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means Clustering on Iris Dataset",fontsize = 12)


figu, axe = plt.subplots(1,2,figsize=(10,8))
axe[0].scatter(X[:,2],X[:,3],c=km.labels_,edgecolor="k",s=50)
axe[0].set_xlabel("Petal Length")
axe[0].set_ylabel("Petal Width")
axe[0].set_title("KMeans Clustering")
axe[1].scatter(X[:,2],X[:,3],c=y,edgecolor="k",s=50)
axe[1].set_xlabel("Petal Length")
axe[1].set_ylabel("Petal Width")
axe[1].set_title("Original Iris Dataset")