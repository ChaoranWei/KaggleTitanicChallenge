from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn.externals import joblib
import numpy as np

training = np.load('pickle/train.pkl')
test = np.load('pickle/test.pkl')

ID = test[:,0]
test = np.delete(test,0,1)

survival = training[:,0]
training = np.delete(training, 0,1)

##################################################################################################################

alldata = ClassificationDataSet(len(training[1]), nb_classes=2)

for n in xrange(len(training)):
    alldata.addSample(training[n], survival[n])

alldata._convertToOneOfMany( )

fnn = buildNetwork( alldata.indim, 5, alldata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=alldata, momentum=0.7, verbose=True, weightdecay=0.001)



##################################################################################################################

pred = []
for i in test:
    softmax = fnn.activate(i)
    for j in range(len(softmax)):
        if softmax[j] == max(softmax):
            pred.append(j)
    print softmax
print pred