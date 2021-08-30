# Load libraries
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print ("Dataset Lenght:: ", len(dataset))
print ("Dataset Shape:: ", dataset.shape)
print ("Dataset:: ")
print(dataset.head())

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                  max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, Y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, Y_train)

clf_gini.predict([[4, 4, 3, 3]])

y_pred = clf_gini.predict(X_test)
y_pred_en = clf_entropy.predict(X_test)

print ("Accuracy is ", accuracy_score(Y_test,y_pred)*100)
print ("Accuracy is ", accuracy_score(Y_test,y_pred_en)*100)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, y_pred_en)
print(confusion_matrix)