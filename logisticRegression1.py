import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
import pickle

#np.set_printoptions(threshold='nan')
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)
print len(test[1])
survival = training[:,0]
training = np.delete(training, 0,1)
print len(training[1])
LR = LogisticRegression(penalty = 'l1', C = 1)

LR.fit(training, survival)

Prediction = LR.predict(test)
Prediction.dump('pickle/predictionLR.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(LR,'pickle/LR.pkl')


