import csv
import numpy as np
import pandas as pd
import pylab as p
import pickle

#numpy
#######################################################################################################
file = csv.reader(open('train.csv'))
notitle = file.next()

dataset = []

for row in file:
    dataset.append(row)

dataset = np.array(dataset)


#Shows the number of passengers
NumPass = np.size(dataset[0::,1].astype(np.float))
#Number of survivors
NumSurv = np.sum(dataset[0::,1].astype(np.float))
#survivalrate

SurvRate = NumSurv / NumPass
#find only male
Male_only = dataset[0::,4] == 'male'
Female_only = dataset[0::,4] != 'male'

MaleData = dataset[Male_only]
FemaleData = dataset[Female_only]

#read test file and predict

test = csv.reader(open('test.csv','rb'))
header = test.next()



predictionFile = open('submission.csv','wb')
prediction = csv.writer(predictionFile)
prediction.writerow(['PassengerId','Survived'])
for person in test:
    if person[3] == 'male':
        prediction.writerow([person[0],'1'])
    else:
        prediction.writerow([person[0],'0'])


predictionFile.close()


#pandas
######################################################################################################################3


pandafile = pd.read_csv('train.csv',header = 0)


#print pandafile.Age[0:10].median()
#print pandafile[['Age','Pclass','Sex']][0:10]
#print pandafile[pandafile['Age'] < 10 ].Pclass
#print pandafile[pandafile['Age'].isnull()][['Age','Sex','Survived']]
#print len(pandafile[(pandafile['Sex'] == 'female') & (pandafile['Pclass'] == 3) & (pandafile['Survived'] == 1)])
#Plot histogram
#pandafile['Age'].hist(bins = 13, range = (0,100),alpha = 0.8)

#p.show()
#data cleaning


df = pandafile
#print df
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
#print(df['Gender'].head(5))         #change it into M and F
df['Gender'] = df['Sex'].map({'male':1,'female':0}).astype(int)
#print(df['Gender'].head(3) )

df['age'] = df['Age']
median_ages = np.zeros((2,3))

for i in range(2):
    for j in range(3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()


for i in range(len(df['age'])):
    if pd.isnull(df['age'][i]):

        df['age'][i] = median_ages[df['Gender'][i],df['Pclass'][i] - 1]

#print(df['age'])

#family size
df['FamilySize'] = df['SibSp'] + df['Parch']

df['Ageclass'] = df['age'] * df['Pclass']

f2 = df['FamilySize'].hist()
#p.show()

#print df.dtypes[df.dtypes.map(lambda x: x == 'object')]
df = df.drop(['Name','Sex','Ticket','Cabin','Embarked','Age','PassengerId'], axis = 1)
#print df
meanlist = []
for i in df:
    meanlist.append(df[i].mean())

training = df.values  #.values transform pandas dataframe into numpy array



#for i in range(len(training)):
#    for j in range(len(training[0])):
#        if j != 0:
#            training[i][j] = training[i][j] / meanlist[j]
#print training[1]

training.dump('pickle/train.pkl')
