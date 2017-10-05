from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

data = numpy.loadtxt("Data/data.csv", delimiter=",")

X = data[:,0:8]
Y = data[:,8]

print X

random_state = numpy.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,random_state=42)

n_feat = X_train.shape[1]
n_targets = y_train.max() + 1

reg = LogisticRegression(C=1e5)

rs = GridSearchCV(reg, param_grid={
    'C': [1],
    'solver':["liblinear"],
    'max_iter':[5000],
    'tol':[1e-8]},verbose=2)

rs.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = y_test
predicted = rs.predict(X_test)

print("Classification report for classifier %s:\n%s\n" % (
    reg, classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))

print rs.best_params_
