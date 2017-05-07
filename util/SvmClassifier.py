from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

class SvmClassifier(object):
    """description of class"""

    def __init__(self):

        self.x_scaler = StandardScaler()
        self.svm = None

    def train(self, X, y, n_splits = 10, test_size = 0.5, random_state=0):

        self.x_scaler.fit(X)
        scaled_X = self.x_scaler.transform(X)

        acc = 0
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        for train_index, test_index in sss.split(scaled_X, y):
            X_train, X_test = scaled_X[train_index], scaled_X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            temp_svm = LinearSVC()
            temp_svm.fit(X_train, y_train)

            score = round(temp_svm.score(X_test, y_test), 4)
            print('Test Accuracy of SVC = ', score)
            if score > acc:
                acc = score
                self.svm = temp_svm

    def predict(self, X):

        scaled_X = self.x_scaler.transform(X)        
        return self.svm.predict(scaled_X)

