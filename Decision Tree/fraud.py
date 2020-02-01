# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:55:07 2019

@author: Gopi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fraud=pd.read_csv('file:///C:/Users/Gopi/Desktop/DS/ASSIGNMENTS/Decision tree/Fraud_check.csv')
labels=['risky','good']
bins=[10000,30000,99619]
fraud['taxincome']=pd.cut(fraud['taxincome'],bins=bins,labels=labels)
fraud.head()


y=fraud.iloc[:,2]
fraud1= fraud.drop('taxincome', axis=1)
fraud2=pd.concat([y,fraud1],axis=1)

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
fraud2['undergrad']=lb_make.fit_transform(fraud['undergrad'])
fraud2['maritalstat']=lb_make.fit_transform(fraud['maritalstat'])
fraud2['urban']=lb_make.fit_transform(fraud['urban'])



colnames=list(fraud2.columns)
predx= colnames[1:5]
predy=colnames[0]

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud2,test_size=0.2)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train[predx],train[predy])

preds=model.predict(test[predx])
pd.Series(preds).value_counts()
pd.crosstab(test[predy],preds)

np.mean(train.taxincome == model.predict(train[predx])) #96 accuracy

np.mean(preds==test.taxincome)#75 accuracy
