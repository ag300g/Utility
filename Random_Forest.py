import csv
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score


data = genfromtxt('../forest_train.csv', delimiter=',')
data_val = genfromtxt('../forest_validation.csv', delimiter=',')

tr_sub = range(0, 1000)
val_sub = range(0, 1000)

X_scaled = preprocessing.scale(data[tr_sub,0:10])
X_bin = data[tr_sub, 10:50]
X_new = np.hstack([X_scaled, X_bin])
y = data[tr_sub, 50]

X_val_scaled = preprocessing.scale(data_val[val_sub,0:10]) 
X_val_bin = data_val[val_sub,10:50]
X_val_new = np.hstack([X_val_scaled,X_val_bin])
y_val = data_val[val_sub, 50]


n = 80
nf = 10
split = 1
clf = RandomForestClassifier(n_estimators=n, max_depth=None,\
                             min_samples_split=split, random_state=0, \
                             max_features= nf)
                             
fit = clf.fit(X_new, y)     

# 3-fold cross validation score                  
scores = cross_val_score(clf, X_new, y).mean()

# accuray on the validation set
val_sc = fit.score(X_val_new, y_val)
