from sklearn import linear_model
import pandas as pd
from sklearn import datasets ## imports datasets from scikit-learn

data = datasets.load_boston() ## loads Boston dataset from datasets library
print(data.DESCR)
print(data.feature_names) #print the column names of the independent variables
print(data.target)

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)

print(predictions[0:5])
print("Score is:", lm.score(X,y))
print("Coefficients are :", lm.coef_)
print("The value of intercept is : ", lm.intercept_)