import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsClassifier
import pickle
import cPickle

#np.set_printoptions(threshold='nan')
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0, 1)

KNN = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', algorithm = 'auto', )

KNN.fit(training, survival)

Prediction = KNN.predict(test)
Prediction.dump('pickle/predictionKNN.pkl')
ID.dump('pickle/ID.pkl')
with open('pickle/knn.pickle', 'wb') as f:
    cPickle.dump(KNN, f)
