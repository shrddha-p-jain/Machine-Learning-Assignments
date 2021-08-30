#Importing Libraries and Loading Datasets
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_iris()

#Creating Our Naive Bayes Model Using Sckit-learn
model = GaussianNB()
model.fit(dataset.data, dataset.target)

#Making Predictions
expected = dataset.target
predicted = model.predict(dataset.data)

#Getting Accuracy and Statistics
print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))