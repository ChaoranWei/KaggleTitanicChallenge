from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier


training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)
print test

survival = training[:,0]
training = np.delete(training, 0,1) #first 0 is the second element, second 1 is column 0 is row

print training

#random forest

ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 30, learning_rate = 1, random_state = None)
#ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 16, learning_rate = 1, random_state = 1)
#max_depth 1 to 4
ada.fit(training, survival)

Prediction = ada.predict(test)
Prediction.dump('pickle/predictionADA.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(ada,'pickle/ada.pkl')