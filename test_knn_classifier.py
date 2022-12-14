from sklearn.neighbors import KNeighborsClassifier as SK_KNeighborsClassifier
from knn import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

X_data, y_data = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

predicted = knn.predict(X_test)

print(classification_report(y_test, predicted))


clf = SK_KNeighborsClassifier()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(classification_report(y_test, predicted))
