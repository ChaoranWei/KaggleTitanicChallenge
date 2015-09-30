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
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
import cPickle


#Note: now we have decent results from SVM (0.77),GB (0.74)
#preparation
######################################################################################################################as
training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')
SVM1 = np.load('pickle/SVM1.pkl')
SVM2 = np.load('pickle/SVM2.pkl')
LR = np.load('pickle/LR.pkl')
GB = np.load('pickle/GB.pkl')
with open('pickle/RF.pkl', 'rb') as f:
    RF = cPickle.load(f)
NB = np.load('pickle/NB.pkl')
LDA = np.load('pickle/LDA.pkl')
with open('pickle/knn.pickle', 'rb') as f:
    KNN = cPickle.load(f)



ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0, 1)

#ensemble structure
########################################################################################################################

class ensemble:
    def __init__(self, TrainInput, TrainOutput, ModelList):
        self._input = TrainInput
        self._output = TrainOutput
        self._model = ModelList
        self._score = None
        self._pred = None

    def MajorityVote(self, test):  #it can be a list of all prediction from test data
      #  Pred = self._GetResult(self._)
        pred = self.predict(test)
        result = []
        for i in range(len(pred[1])):
            count = 0
            for sublist in pred:
                temp = round(sublist[i])
                if temp == 0:
                    count = count + 1

            if count >= float(len(pred)) / 2: #if there is a tie, I choose 0 because there are more people die than survived
                result.append(0)
            else:
                result.append(1)
        return result

    def LinearCombination(self, ListParam, test):
        '''Linear combanition of all ensemble models to return prediction from test data
        List: list of all predictions
        ListParam: weight of each model
        '''
        pred = self.predict(test)
        result = []
        for i in range(len(pred[0])):
            count = 0
            for j in range(len(pred)):
                count = count + pred[j][i] * ListParam[j]
            result.append(int(round(count)))
        return result

    def predict(self, test):
        '''Get a list of prediction for each model
        '''
        Pred = []
        for model in self._model:
            ModelList = []
            model.fit(self._input, self._output)
            prediction = model.predict_proba(test)
            for j in range(len(prediction)):
                ModelList.append(prediction[j][1])
            Pred.append(ModelList)
        self._pred = Pred
        return Pred


    def score(self, ValIn, ValOut, option, ListParam = None):
        '''Take on validation prediction to compare with actual output.
         for option: 1 for majority vote,
                     2 for linear combination
        ListParam only used for linear combination
         '''
        if option == 1:
            prediction = self.MajorityVote(ValIn)
        elif option == 2:
            prediction = self.LinearCombination(ListParam, ValIn)

        count = 0
        for i in range(len(ValIn)):
            if prediction[i] == ValOut[i]:
                count = count + 1
        precision = float(count) / len(ValIn)
        return precision


#actual models
######################################################################################################################3

#BaggingKNN
bagging = BaggingClassifier( KNeighborsClassifier(n_neighbors = 10, weights = 'distance', algorithm = 'auto', ),
                            n_estimators = 30, max_samples=0.5, max_features=0.5)
bagging.fit(training, survival)
Prediction = bagging.predict(test)

Prediction.dump('pickle/predictionBagKNN.pkl')
ID.dump('pickle/ID.pkl')
joblib.dump(bagging,'pickle/BKNN.pkl')

#majority vote example1

ensemble1 = ensemble(training,survival, [SVM1, SVM2, LR, GB])
accuracy = ensemble1.LinearCombination([0.25, 0.25, 0.2,0.3], test)

prediction = np. asarray(accuracy)
prediction.dump('pickle/predictionE1.pkl')


#run and test
#####################################################################################################################

