# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) # ColumnTransfer in next version
X = onehotencoder.fit_transform(X).toarray() # Produces dummy variables

# Avoiding the Dummy Variable Trap [we need just one dummy less]
# Actually sklearn library takes care of dummy variables & we don't need to do it manually
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# sklearn takes care of 'Feature Scaling' automatically

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ***** Predicting the Test set results *****
y_pred = regressor.predict(X_test)

# Building an optimal model using Backward Elimination
import statsmodels.formula.api as sm
# Adding a column of Ones to X as [x^0 = 1] for b0 [in b0 x^0] just for statsmodels
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # or addconst?
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Remove x2: index 2 [too high]
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Remove x1: index 1 [too high]
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Remove x2: index 2 [more than 5%]
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Looks OK Now --> but one more time!

# Remove x2: index 2 [slightly more than 5%]
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

