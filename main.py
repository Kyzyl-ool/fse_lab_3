import numpy as np
import useful_package
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = np.arange(1, 100, 0.5)


Y_poly = useful_package.polynom_3(X)
Y_hyper = useful_package.hyperbola(X)

X_poly_train, X_poly_test, Y_poly_train, Y_poly_test = train_test_split(X, Y_poly, test_size=0.5, shuffle=True)
X_hyper_train, X_hyper_test, Y_hyper_train, Y_hyper_test = train_test_split(X, Y_hyper, test_size=0.5, shuffle=True)

model_poly = RandomForestRegressor()
model_poly.fit(X_poly_train.reshape(-1, 1), Y_poly_train)

model_hyper = RandomForestRegressor()
model_hyper.fit(X_hyper_train.reshape(-1, 1), Y_hyper_train)

mse_poly = ((model_poly.predict(X_poly_test.reshape(-1, 1)) - Y_poly_test)**2).mean()

mse_hyper = ((model_hyper.predict(X_hyper_test.reshape(-1, 1)) - Y_hyper_test)**2).mean()

print('MSE for poly:', mse_poly)
print('MSE for hyper:', mse_hyper)
