from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.
    Parameters
    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.
    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.
        Parameters
        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels
        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        X : numpy array, shape = [n_samples, n_features]
        Returns
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule
        """
        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):
        """
        Parameters
        X : numpy array, shape = [n_samples, n_features]
        Returns
        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.
        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

data = numpy.loadtxt("Data/data.csv", delimiter=",")

X = data[:,0:8]
Y = data[:,8]

print X

random_state = numpy.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25,random_state=random_state)

n_feat = X_train.shape[1]
n_targets = y_train.max() + 1


bayes = GaussianNB()
reg = LogisticRegression(C=1,max_iter=5000,tol=1e-08,solver='liblinear')
grad_boost = GradientBoostingClassifier(max_features='log2',loss='exponential',learning_rate=0.1,n_estimators=100,max_depth=30)
forest = RandomForestClassifier(max_features='log2',n_estimators=200, criterion='entropy',max_depth=20)
ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
svm = SVC(C=1,kernel='linear',gamma=0.1,probability=True)

clf = EnsembleClassifier(clfs=[bayes, reg, grad_boost, forest, ada_boost, svm],weights=[2,1,1,2,1,1])

clf.fit(X_train,y_train)
             
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = y_test
predicted = clf.predict(X_test)

print("Classification report for classifier %s:\n%s\n" % (
    clf, classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
