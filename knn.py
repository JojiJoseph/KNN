from email.policy import default
import numpy as np
from collections import defaultdict


class KNeighborsClassifier:
    def __init__(self, n_neighbours: int = 5) -> None:
        """Constructor of KNN classifier

        Parameters
        ----------
        n_neighbours : int, optional
            number of neighbours, by default 5
        """
        self.n_neighbours = n_neighbours

    def fit(self, X_train, y_train):
        """Fits KNN classifier from training dataset.

        Parameters
        ----------
        X_train : _type_
            Trainin data
        y_train : _type_
            Target labels
        """
        self.size = len(X_train)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predicts the class labels for the input data.

        Parameters
        ----------
        X_test : _type_
            input data

        Returns
        -------
        _type_
            prediction
        """
        predicted = []
        for X in X_test:
            dists = np.linalg.norm(self.X_train - X, axis=-1)
            dists_with_label = list(zip(dists, self.y_train))
            dists_with_label.sort(key=lambda x: x[0])
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


class KNeighborsRegressor:
    def __init__(self, n_neighbours=5) -> None:
        """Constructor of KNN regressor

        Parameters
        ----------
        n_neighbours : int, optional
            number of neighbours, by default 5
        """
        self.n_neighbours = n_neighbours

    def fit(self, X_train, y_train):
        """Fits KNN regressor from training dataset.

        Parameters
        ----------
        X_train : _type_
            Trainin data
        y_train : _type_
            Target values
        """
        self.size = len(X_train)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predicts the output values for the input data.

        Parameters
        ----------
        X_test : _type_
            input data

        Returns
        -------
        _type_
            prediction
        """
        predicted = []
        for X in X_test:
            dists = np.linalg.norm(self.X_train - X, axis=-1).tolist()
            dists_with_label = list(zip(dists, self.y_train))
            dists_with_label.sort(key=lambda x: x[0])
            predicted.append(
                np.mean([label for _, label in dists_with_label[:self.n_neighbours]]))
        return np.array(predicted)
