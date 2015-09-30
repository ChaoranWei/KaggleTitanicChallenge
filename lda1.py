import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.lda import LDA
from sklearn.externals import joblib



training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)
print test

survival = training[:,0]
training = np.delete(training, 0,1) #first 0 is the second element, second 1 is column 0 is row

print training

#random forest

ld = LDA(solver = 'svd', shrinkage = None)
ld.fit(training, survival)

Prediction = ld.predict(test)
Prediction.dump('pickle/predictionLDA.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(ld,'pickle/LDA.pkl')