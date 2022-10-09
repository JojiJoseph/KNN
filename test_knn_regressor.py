import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_data = np.linspace(0,10,100)
y_data = np.sin(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))

from knn import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbours=5)
knn.fit(X_train, y_train)

predicted = knn.predict(X_test)

plt.scatter(X_train.flatten(), y_train.flatten(), label="Training data")
plt.scatter(X_test.flatten(), predicted.flatten(), label="Testing data")
plt.legend()
plt.show(block=False)

from sklearn.neighbors import KNeighborsRegressor

plt.figure()
reg = KNeighborsRegressor()
reg.fit(X_train, y_train)
predicted = reg.predict(X_test)

plt.scatter(X_train.flatten(), y_train.flatten(), label="Training data")
plt.scatter(X_test.flatten(), predicted.flatten(), label="Testing data")
plt.legend()
plt.show()

