import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib

from sklearn.ensemble import GradientBoostingClassifier
import pickle

#np.set_printoptions(threshold='nan')
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0,1) #first 0 is the second element, second 1 is column 0 is row


#random forest

#GB = GradientBoostingClassifier(loss = 'exponential', learning_rate = 1,n_estimators= 26, max_depth = 2)
GB = GradientBoostingClassifier(loss = 'deviance', learning_rate = 1,n_estimators= 32, max_depth = 1)
GB.fit(training, survival)

Prediction = GB.predict(test)
Prediction.dump('pickle/predictionGB.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(GB,'pickle/GB.pkl')