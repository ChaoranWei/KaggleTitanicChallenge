import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from ensemble import ensemble
import cPickle

training = np.load('pickle/train.pkl')

survival = training[:,0]
training = np.delete(training, 0,1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(training, survival, test_size=0.3)

########################################################################################################################

#random forest
#score = []


#for i in range(1,10):
#    print i
#    RF = RandomForestClassifier(n_estimators = 13, criterion = 'gini', max_features = 'log2', warm_start = False, max_depth = 5,class_weight = 'auto', random_state = 10)
#    RF.fit(X_train, y_train)


#    score.append(RF.score(X_test, y_test))

#plt.plot(score)
#plt.show()


#logistic regression

#Score = []
#for i in range(10,100):
#    print i
#    kf = cross_validation.KFold(len(training),10 ,shuffle = True, random_state = 55)
#    score = 0
#    for train, test in kf:
#        X_train, X_test = training[train],training[test]
#        y_train, y_test = survival[train],survival[test]
#        LR = LogisticRegression(penalty = 'l1', C = 20)
#        LR.fit(X_train, y_train)
#        score = score + LR.score(X_test, y_test)

#    Score.append(score / 10.0)
#plt.plot(Score)
#plt.show()

#SVM
#score = []

#for i in range(30,70):
#    print(i)
#    SVM = svm.SVC(C =  40, kernel = 'rbf',gamma = 0.007)
    #SVM = svm.LinearSVC(C = 0.1 * i, penalty = 'l2' ) do not use it, oslate like crazy
#    SVM = svm.NuSVC(nu = 0.43, kernel = 'rbf', degree = 3, gamma = 0.0, probability = True)
#    SVM.fit(X_train, y_train)

 #   score.append(SVM.score(X_test, y_test))

#plt.plot(score)
#plt.show()

#KNN

#score = []

#for i in range(1,100):
#    print(i)
#    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'auto', )
#    print('con')
#    knn.fit(X_train, y_train)
#    print('train')

#    score.append(knn.score(X_test, y_test))

#plt.plot(score)
#plt.show()

#Naive Bayes
#score = []

#nb = GaussianNB()
#nb.fit(X_train, y_train)

#print nb.score(X_test, y_test)

#decision tree

#DT = DecisionTreeClassifier()
#DT.fit(X_train, y_train)

#print DT.score(X_test, y_test)


#LDA
#score = []

#for i in range(1,100):
#    ld = LDA(solver = 'svd', shrinkage = None)
#    ld.fit(X_train, y_train)
#    score.append(ld.score(X_test, y_test))

#plt.plot(score)
#plt.show()

#Adaboost

#score = []

#for i in range(1,100):
#    print i
#    ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 30, learning_rate = 1, random_state = None)
 #   ada.fit(X_train, y_train)

#    score.append(ada.score(X_test, y_test))

#plt.plot(score)
#plt.show()
# kf = cross_validation.KFold(len(training),10 ,shuffle = True, random_state = 25)
#    score = 0
#    for train, test in kf:
#        X_train, X_test = training[train],training[test]
#        y_train, y_test = survival[train],survival[test]
#        GB = GradientBoostingClassifier(loss = 'exponential', learning_rate = 1,n_estimators= 32, max_depth = 1)
#        GB.fit(X_train, y_train)
#        score = score + GB.score(X_test, y_test)

#    Score.append(score / 10.0)

#k-folds
#Score = []
#for i in range(1,40):
#    print i
#    kf = cross_validation.KFold(len(training),10,shuffle = True, random_state = 25)
#    score = 0
#    for train, test in kf:
#        X_train, X_test = training[train],training[test]
#        y_train, y_test = survival[train],survival[test]
#        ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = i, learning_rate = 1, random_state = 1)
#        ada.fit(X_train, y_train)
#        score = score + ada.score(X_test, y_test)
#
#    Score.append(score / 10.0)
#plt.plot(Score)
#plt.show()

#bootstrapping This method yields unexpectedly low score
#bs = cross_validation.Bootstrap(len(training), 5, 0.6, 0.3, random_state = 28)
#score = 0
#for train, test in bs:
#    X_train, X_test = training[train],training[test]
#    y_train, y_test = survival[train],survival[test]
#    ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 50, learning_rate = 1, random_state = 1)
#    ada.fit(X_train, y_train)
#    score = score + ada.score(X_test, y_test)

#print score / 10.0


#gradient boosting
#Score = []
#for i in range(1,20):
#    print i
#    kf = cross_validation.KFold(len(training),10 ,shuffle = True, random_state = 10)
#    score = 0
#    for train, test in kf:
#        X_train, X_test = training[train],training[test]
#        y_train, y_test = survival[train],survival[test]
#        GB = GradientBoostingClassifier(loss = 'exponential', learning_rate = 1,n_estimators= 26, max_depth = 2)
#        GB.fit(X_train, y_train)
#        score = score + GB.score(X_test, y_test)

#    Score.append(score / 10.0)
#plt.plot(Score)
#plt.show()

#bagging with KNN
#Score = []
#for i in range(10,20):
#    print i
#    kf = cross_validation.KFold(len(training),10 ,shuffle = True)#, random_state = 25)
#    score = 0
#    for train, test in kf:
#        X_train, X_test = training[train],training[test]
#        y_train, y_test = survival[train],survival[test]
#        bagging = BaggingClassifier( KNeighborsClassifier(n_neighbors = 10, weights = 'distance', algorithm = 'auto', ),
#                            n_estimators = 30, max_samples=0.5, max_features=0.5)
#        bagging.fit(training, survival)
#        score = score + bagging.score(X_test, y_test)

#    Score.append(score / 10.0)
#plt.plot(Score)
#plt.show()

##########################################################################################################################

#ensemble validation 1

SVM = np.load('pickle/SVM.pkl')
LR = np.load('pickle/LR.pkl')
GB = np.load('pickle/GB.pkl')
with open('pickle/RF.pkl', 'rb') as f:
    RF = cPickle.load(f)
NB = np.load('pickle/NB.pkl')
LDA = np.load('pickle/LDA.pkl')
with open('pickle/knn.pickle', 'rb') as f:
    KNN = cPickle.load(f)



#print X_train[1]
#print y_train[1]
#Accuracy = 0
#for i in range(10):
#    ensemble1 = ensemble(X_train,y_train, [SVM, RF, LR, GB, LDA])
#    accuracy = ensemble1.score(X_test, y_test, 2, [0.25, 0.15, 0.25,0.2,0.15])
#    Accuracy = accuracy + Accuracy

#print Accuracy
