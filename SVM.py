
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn import svm 
from sklearn.cross_validation import cross_val_score

data = genfromtxt('../forest_train.csv', delimiter=',')
data_val = genfromtxt('../forest_validation.csv', delimiter=',')

tr_sub = range(0, 1000)
val_sub = range(0, 1000)

X_scaled = preprocessing.scale(data[tr_sub,0:10])
X_new = np.hstack([X_scaled, data[tr_sub, 10:50]])
y = data[tr_sub, 50]

X_val_scaled = preprocessing.scale(data_val[val_sub,0:10]) 
X_val_new = np.hstack([X_val_scaled, data_val[val_sub,10:50]])
y_val = data_val[val_sub, 50]

rbf_clf = svm.SVC(C=7, kernel = 'rbf', gamma=0.2)
scores = cross_val_score(rbf_clf, X_new, y).mean()

rbf_fit = rbf_clf.fit(X_new,y)
ypred_rbf = rbf_fit.score(X_new, y)
ypred_val_rbf = rbf_fit.score(X_val_new, y_val)
