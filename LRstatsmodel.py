import pandas as pd
import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn

data = datasets.load_boston() ## loads Boston dataset from datasets library 

print(data.DESCR)
print(data.feature_names) #print the column names of the independent variables
print(data.target) #print the column values of the dependent variables


# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df)
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df["RM"] ## X usually means our input variables (or independent variables)
y = target["MEDV"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

"""First we have what’s the dependent variable and the model and the method. 
OLS stands for Ordinary Least Squares and the method “Least Squares” means that we’re trying 
to fit a regression line that would minimize the square of distance from the regression line"""

X = df[["RM", "LSTAT"]]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())

"""Interpreting the Output — We can see here that this model has a much higher 
R-squared value — 0.948, meaning that this model explains 94.8% of the variance in 
our dependent variable. Whenever we add variables to a regression model, R² will be
 higher, but this is a pretty high R². We can see that both RM and LSTAT are statistically 
 significant in predicting (or estimating) the median house value; not surprisingly , 
 we see that as RM increases by 1, MEDV will increase by 4.9069 and when LSTAT increases 
 by 1, MEDV will decrease by -0.6557. As you may remember, LSTAT is the percentage of lower 
 status of the population, and unfortunately we can expect that it will lower the median 
 value of houses. With this same logic,the more rooms in a house, usually the higher its value 
 will be."""