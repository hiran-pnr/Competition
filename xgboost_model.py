# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:04:07 2022

@author: Lenovo
"""

import pandas as pd
import numpy as np
data = pd.read_csv('employee_promotion.csv')
data.drop('employee_id', axis=1, inplace=True)
data['education']=data['education'].fillna(data['education'].mode()[0])
data['avg_training_score']=data['avg_training_score'].fillna(data['avg_training_score'].mode()[0])
data['previous_year_rating']=data['previous_year_rating'].fillna(value=0)
Q1=np.percentile(data['length_of_service'], 25, interpolation='midpoint')
Q2=np.percentile(data['length_of_service'], 50, interpolation='midpoint')
Q3=np.percentile(data['length_of_service'], 75, interpolation='midpoint')
IQR = Q3-Q1
upper_limit=round(Q3+(1.5 * IQR), 4)
lower_limit=round(Q1-(1.5 * IQR), 4)
outlier=[]
for x in data['length_of_service']:
    if (x<lower_limit) or (x>upper_limit):
        outlier.append(x)
data['length_of_service']=np.where(data['length_of_service']>upper_limit,upper_limit,np.where(data['length_of_service']<lower_limit,lower_limit,data['length_of_service']))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data[['education']] = data[['education']].apply(le.fit_transform)
data[['gender']] = data[['gender']].apply(le.fit_transform)
data[['recruitment_channel']] = data[['recruitment_channel']].apply(le.fit_transform)
data.drop(['department', 'region'],axis=1, inplace=True)
y = data['is_promoted']
X= data.drop(['is_promoted'], axis=1)
min_max = preprocessing.MinMaxScaler()
X = min_max.fit_transform(X)
X = pd.DataFrame(X)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_up, y_up = smote.fit_resample(X, y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_up, y_up, random_state=42, test_size=0.25)
X_train=X_train.values
X_test=X_test.values
import xgboost as xgb
XGB4 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=250, max_depth=10,
                        min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-05,
                        use_label_encoder=False)
XGB4.fit(X_train, y_train)
import pickle
pickle.dump(XGB4, open('model.pkl', 'wb'))

