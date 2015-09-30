import csv
import pickle
import numpy as np
Prediction = np.load('pickle/predictionE1.pkl')
print Prediction
ID = np.load('pickle/ID.pkl')


predictionFile = open('submission.csv','wb')
prediction = csv.writer(predictionFile)
prediction.writerow(['PassengerId','Survived'])
for i in range(len(ID)):
    id = int(ID[i])
    survived = int(Prediction[i])
    prediction.writerow([id,survived])

predictionFile.close()

