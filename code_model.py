# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:53:39 2021

@author: Admin
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv('flight_deploy_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

X = df.drop(['Price'],axis=1)
y = df['Price']

#split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=50)

from catboost import CatBoostRegressor

cat = CatBoostRegressor()
cat.fit(X_train,y_train)

y_predict = cat.predict(X_test)
