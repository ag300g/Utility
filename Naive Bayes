import csv
import numpy as np
from operator import add
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def predict(x1, x2, y, fit1, fit2):
    prob1 = fit1.predict_proba(x1)
    prob2 = fit2.predict_proba(x2)
    
    prob = map(lambda x: list(x/2), map(add, prob1, prob2))
    ypred = np.array(map(lambda x: x.index(max(x))+1, prob))
    return(accuracy_score(ypred, y))

    
    
data = genfromtxt('../forest_train.csv', delimiter=',')
data_val = genfromtxt('../forest_validation.csv', delimiter=',')

tr_sub = range(0, 1000)
val_sub = range(0, 1000)
X_cont = preprocessing.scale(data[tr_sub,0:10])
X_bin = data[tr_sub, 10:50]
y = data[tr_sub, 50]

X_val_cont = preprocessing.scale(data_val[val_sub,0:10]) 
X_val_bin = data_val[val_sub,10:50]
y_val = data_val[val_sub, 50]

clf_cont = GaussianNB()
fit_cont = clf_cont.fit(X_cont, y)

clf_bin = BernoulliNB()
fit_bin = clf_bin.fit(X_bin, y)

pred = predict(X_val_cont, X_val_bin, y_val, fit_cont, fit_bin)
