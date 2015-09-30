import csv
import numpy as np
import pandas as pd
import pylab as p
import pickle
from sklearn import linear_model


pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
#Getting started with data
pandafile = pd.read_csv('test.csv',header = 0)


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
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
#print(df['Gender'].head(5))         #change it into M and F
df['Gender'] = df['Sex'].map({'male':1,'female':0}).astype(int)
#print(df['Gender'].head(3) )

#add military title
#df['military'] = 0
#df['Richman'] = 0
#df['Richlady'] = 0
#print df['military']
#count = 0
#for i in range(len(df['Name'])):
#    if 'Mme' in df['Name'][i] or 'Mlle' in df['Name'][i]:
#        df['military'][i] = 1

#    elif 'Capt' in df['Name'][i] or 'Don' in df['Name'][i] or 'Major' in df['Name'][i] or 'Sir' in df['Name'][i]:
#        df['Richman'][i] = 1

#    elif 'Dona' in df['Name'][i] or 'Lady' in df['Name'][i] or 'the Countess' in df['Name'][i] or 'Jonkheer' in df['Name'][i]:
#        df['Richlady'][i] = 1
#        count = count + 1

df['age'] = df['Age']
median_ages = np.zeros((2,3))

for i in range(2):
    for j in range(3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

for i in range(len(df['age'])):
    if pd.isnull(df['age'][i]):

        df['age'][i] = median_ages[df['Gender'][i],df['Pclass'][i] - 1]

#family size
df['FamilySize'] = df['SibSp'] + df['Parch']

#additional features
df['Ageclass'] = df['age'] * df['Pclass']
#df['agefare'] = df['age'] * df['Fare']
#df['familyfare'] = df['FamilySize'] * df['Fare']
#df['familyage'] = df['FamilySize'] * df['age']
#df['familyclass'] = df['FamilySize'] * df['Pclass']

f2 = df['FamilySize'].hist()
#p.show()
print df['Name']

#print df.dtypes[df.dtypes.map(lambda x: x == 'object')]
df = df.drop(['Name','Sex','Ticket','Cabin','Embarked','Age'], axis = 1)

meanlist = []
for i in df:
    meanlist.append(df[i].mean())
training = df.values    #.values transform pandas dataframe into numpy array

#check if there is nan remained
for i in range(len(training)):
    for j in range(len(training[1])):
        if np.isnan(training[i][j]):
            training[i][j] = 0 # specific assignment, cannot generalize

#for i in range(len(training)):
#    for j in range(len(training[0])):
#        if meanlist[j] != 0 and j != 0:
#            training[i][j] = training[i][j] / meanlist[j]
#print training[1]



training.dump('pickle/test.pkl')