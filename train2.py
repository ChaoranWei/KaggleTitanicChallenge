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
print(median_ages)

for i in range(len(df['age'])):
    if pd.isnull(df['age'][i]):

        df['age'][i] = median_ages[df['Gender'][i],df['Pclass'][i] - 1]

#print(df['age'])

#add military title
df['military'] = 0
df['Richman'] = 0
df['Richlady'] = 0
#print df['military']

for i in range(len(df['Name'])):
    if 'Mme' in df['Name'][i] or 'Mlle' in df['Name'][i]:
        df['military'][i] = 1
    elif 'Capt' in df['Name'][i] or 'Don' in df['Name'][i] or 'Major' in df['Name'][i] or 'Sir' in df['Name'][i]:
        df['Richman'][i] = 1
    elif 'Dona' in df['Name'][i] or 'Lady' in df['Name'][i] or 'the Countess' in df['Name'][i] or 'Jonkheer' in df['Name'][i]:
        df['Richlady'][i] = 1

#family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['Ageclass'] = df['age'] * df['Pclass']
df['agefare'] = df['age'] * df['Fare']

f2 = df['FamilySize'].hist()
#p.show()

#print df.dtypes[df.dtypes.map(lambda x: x == 'object')]
df = df.drop(['Name','Sex','Ticket','Cabin','Embarked','Age','PassengerId'], axis = 1)
print df
training = df.values    #.values transform pandas dataframe into numpy array

training.dump('pickle/train.pkl')
