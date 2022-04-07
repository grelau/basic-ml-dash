import numpy as np 


x_train = np.random.randn(100, 2)
y_train = 0.2 * x_train[:, 0] - 1.2 * x_train[:, 1]

## train machine model : linear regression

from sklearn.linear_model import LinearRegression 

model = LinearRegression()

model.fit(x_train, y_train)

import joblib 
joblib.dump(model, "linear_regression.joblib")