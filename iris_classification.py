from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target


print("Feature names:", iris.feature_names)
print("Target class labels:", iris.target_names)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("KNN Accuracy:", knn.score(X_test, y_test))


# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", dt.score(X_test, y_test))


# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
print("Logistic Regression Accuracy:", lr.score(X_test, y_test))


# SVM
svm = SVC()
svm.fit(X_train, y_train)
print("SVM Accuracy:", svm.score(X_test, y_test))


# Predict a new flower using SVM (you can try with others too)
sample = [[6.7,9.8,4.5,3.2]]
prediction = svm.predict(sample)
prediction1 = knn.predict(sample)
print("Predicted class:", iris.target_names[prediction][0])
print("Predicted class:", iris.target_names[prediction1][0])
