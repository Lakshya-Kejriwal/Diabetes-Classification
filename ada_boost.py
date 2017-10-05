from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

data = numpy.loadtxt("Data/data.csv", delimiter=",")

X = data[:,0:8]
Y = data[:,8]

print X

random_state = numpy.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,random_state=42)

n_feat = X_train.shape[1]
n_targets = y_train.max() + 1

ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)

rs = GridSearchCV(ada_boost, param_grid={
    'algorithm': ["SAMME","SAMME.R"],
    'learning_rate':[0.01,0.1,1],
    'n_estimators':[200,500]},verbose=2,n_jobs=2)

rs.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = y_test
predicted = rs.predict(X_test)

print("Classification report for classifier %s:\n%s\n" % (
    ada_boost, classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))

print rs.best_params_
