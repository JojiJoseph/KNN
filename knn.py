from email.policy import default
import numpy as np
from collections import defaultdict

class KNeighborsClassifier:
    def __init__(self,n_neighbours=5) -> None:
        self.n_neighbours = n_neighbours
    def fit(self,X_train, y_train):
        self.size = len(X_train)
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predicted = []
        for X in X_test:
            dists = np.linalg.norm(self.X_train - X, axis=-1)
            dists_with_label = list(zip(dists, self.y_train))
            dists_with_label.sort(key=lambda x:x[0])
            # print(dists.shape)
            labels = defaultdict(int)
            for i in range(self.n_neighbours):
                labels[dists_with_label[i][1]] += 1
            best_label = None
            best_cnt = 0
            for label in labels:
                if labels[label] > best_cnt:
                    best_cnt = labels[label]
                    best_label = label
            predicted.append(best_label)
        return np.array(predicted)

