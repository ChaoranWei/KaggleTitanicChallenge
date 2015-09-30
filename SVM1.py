import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib

from sklearn import svm
import pickle

#np.set_printoptions(threshold='nan')
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0, 1)

SVM1 = svm.SVC(C =  40, kernel = 'rbf',gamma = 0.007, probability = True)
SVM2 = svm.NuSVC(nu = 0.43, kernel = 'rbf', degree = 3, gamma = 0.0, probability = True)
SVM1.fit(training, survival)
SVM2.fit(training, survival)

Prediction1 = SVM1.predict(test)
Prediction2 = SVM2.predict(test)

Prediction1.dump('pickle/predictionSVM1.pkl')
Prediction2.dump('pickle/predictionSVM2.pkl')
ID.dump('pickle/ID.pkl')

joblib.dump(SVM1,'pickle/SVM1.pkl')
joblib.dump(SVM2,'pickle/SVM2.pkl')

