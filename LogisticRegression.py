from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

#print(iris)
print(list(iris.keys()))
print(iris.feature_names)
print(iris.data)
print(iris.target_names)

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, test_size=.4)

logi = LogisticRegression()
logi.fit(X_train, y_train)
print(logi.score(X_train, y_train))
print(logi.score(X_test, y_test))
y_pred = logi.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)