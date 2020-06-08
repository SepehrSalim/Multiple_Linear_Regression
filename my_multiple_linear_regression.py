# Multiple Linear Regression

# Importing the libraries
import numpy as np
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

# *** sklearn takes care of 'Feature Scaling' automatically ***

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ***** Predicting the Test set results & showing R-Squared *****
y_pred = regressor.predict(X_test)
r_squared = regressor.score(X_test, y_test)

# <<< Building an optimal model using Backward Elimination by statsmodels >>>
from statsmodels.formula.api import OLS
from statsmodels.tools.tools import add_constant

# Adding a column of Ones to X : add_constant method
X = add_constant(X)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # Adj. R-squared = 0.945

# Let's compare P to 0.05
# Remove x2 (index 2) [0.99 >> 0.05]
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # Adj. R-squared = 0.946

# Remove x1 (index 1) [0.94 >> 0.05]
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # Adj. R-squared = 0.948

# Remove x2 (index 2) [0.602 >> 0.05]
X_opt = X[:, [0, 3, 5]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # Adj. R-squared = 0.948 *** The Best Model!
# Looks Qite Good! Model based on [R&D Spend & Marketing Spend]

# Let's do it one more time!
# Remove x2 (index 2) [0.06 > 0.05] Not very far
X_opt = X[:, [0, 3]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # Adj. R-squared = 0.945 * Decreasing

# *** Adj. R-squared is decreasing now, So the previous model is the best ***

# <<< Now, Prediction Based on This Model! >>>
X_opt = X[:, [0, 3, 5]]
regressor_OLS = OLS(endog = y, exog = X_opt).fit()

X_OLS_test = X_test[:, [2,4]] 
X_OLS_test = add_constant(X_OLS_test)
y_OLS_pred = regressor_OLS.predict(X_OLS_test)

#regressor_OLS.score(y_test)