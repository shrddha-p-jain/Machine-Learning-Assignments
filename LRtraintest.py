import pandas as pd

from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4)

print ("The dimensions of the training data are :",X_train.shape, y_train.shape)
print ("The dimensions of the test data are :",X_test.shape, y_test.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

print ("The Model Score is:", model.score(X_test, y_test))

