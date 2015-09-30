from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib

training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0, 1)

nb = GaussianNB()
nb.fit(training, survival)

Prediction = nb.predict(test)
Prediction.dump('pickle/predictionNB.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(nb,'pickle/NB.pkl')
