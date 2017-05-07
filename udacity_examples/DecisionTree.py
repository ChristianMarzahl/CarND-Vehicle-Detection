import numpy as np
from sklearn import tree


X = [[0, 0], [1, 1]]
Y = [0, 1]
clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf2 = clf2.fit(X, Y)

