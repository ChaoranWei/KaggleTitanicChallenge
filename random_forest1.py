import csv
import numpy as np
import pandas as pd
import pylab as p
from sklearn.externals import joblib
import cPickle

from sklearn.ensemble import RandomForestClassifier
import pickle

#np.set_printoptions(threshold='nan')
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)
#print test

survival = training[:,0]
training = np.delete(training, 0,1) #first 0 is the second element, second 1 is column 0 is row

#print training

#random forest
RF = RandomForestClassifier(n_estimators = 13, criterion = 'gini', max_features = 'log2', warm_start = False, max_depth = 5,class_weight = 'auto', random_state = 10)

#RF = RandomForestClassifier(n_estimators=10)

RF.fit(training, survival)

Prediction = RF.predict(test)
print Prediction
Prediction.dump('pickle/predictionRF.pkl')
ID.dump('pickle/ID.pkl')

with open('pickle/RF.pkl', 'wb') as f:
    cPickle.dump(RF, f)

##############################################################################################################


#ways to improve the model:
#1. most important: cross-validation!!!!
#2. feature engineering, creating more features
#3. cleaning data in a better way
#5. ensemble training
#6.psedu labeling
#7.test all methods using k-fold



#record: random forest(10), age according to Pclass, have family size and age*class: 2860/3034
#logistic regression C =100000 2697/3034 0.75598
#logistic regression C = 1 1906/3045 0.77512
#SVM C = 10000/22, gamma = 0.0006, rbf 0.7703
#naive bayes: 0.70
#decision tree, get a test score of 0.98, and submission score as 0.67, which means it overfits like epic
#LDA with SVD 0.76
#adaboost base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 16, learning_rate = 1, random_state = 1 0.76
#GB = GradientBoostingClassifier(loss = 'deviance', learning_rate = 1,n_estimators= 32, max_depth = 1) 0.7715
#bagging with knn:bagging = BaggingClassifier( KNeighborsClassifier(n_neighbors = 10, weights = 'distance', algorithm = 'auto', ),n_estimators = 30, max_samples=0.5, max_features=0.5)
#0.67 overfit like epic
#neural net, extremely bad
#SVM1: 77512
#SVM2 770

#tomorrow:
# combine pca with other algorithms
#get a better svm

#I might need to use a easier method and do more feature engineering, because 1. score does not improve very much, so it might not be a problem of algorithms
#2.dataset is small so it will be very easy to overfit with fancy algorithms


#strategy: lady alive, weight biase down