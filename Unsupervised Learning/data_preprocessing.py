# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:21:01 2018

Data Preprocessing

@author: Jing
"""

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def getEcoliX():
    ecoli = arff.loadarff('Ecoli-train.arff')
    df = pd.DataFrame(ecoli[0])
    X = df.iloc[:,:7]
    X = StandardScaler().fit_transform(X)
     
    return X

def getEcoliY():
    ecoli = arff.loadarff('Ecoli-train.arff')
    df = pd.DataFrame(ecoli[0])
    Y = df.iloc[:,-1]
    Y = LabelEncoder().fit_transform(Y) 
    return Y

def getEcoliTestX():
    ecoli = arff.loadarff('Ecoli-test.arff')
    df = pd.DataFrame(ecoli[0])
    X = df.iloc[:,:7]
    X = StandardScaler().fit_transform(X)
     
    return X

def getEcoliTestY():
    ecoli = arff.loadarff('Ecoli-test.arff')
    df = pd.DataFrame(ecoli[0])
    Y = df.iloc[:,-1]
    Y = LabelEncoder().fit_transform(Y) 
    return Y

def getAdultX():
    adult = arff.loadarff('Adult-train.arff')
    df2 = pd.DataFrame(adult[0])
    X = df2.iloc[:,df2.columns != 'class']
    column_list = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    X = pd.get_dummies(X, columns=column_list)
    X['education'] = LabelEncoder().fit_transform(X['education'])
    X = X.drop(["workclass_b'?'", "occupation_b'?'"], axis=1) 
    return X

def getAdultY():
    adult = arff.loadarff('Adult-train.arff')
    df2 = pd.DataFrame(adult[0])
    Y = df2.iloc[:, -1]
    Y = LabelEncoder().fit_transform(Y) 
    return Y


def getAdultTestX():
    adult = arff.loadarff('Adult-test.arff')
    df2 = pd.DataFrame(adult[0])
    X = df2.iloc[:,df2.columns != 'class']
    column_list = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    X = pd.get_dummies(X, columns=column_list)
    X['education'] = LabelEncoder().fit_transform(X['education'])
    X = X.drop(["workclass_b'?'", "occupation_b'?'"], axis=1) 
    X = StandardScaler().fit_transform(X)
    return X

def getAdultTestY():
    adult = arff.loadarff('Adult-test.arff')
    df2 = pd.DataFrame(adult[0])
    Y = df2.iloc[:, -1]
    Y = LabelEncoder().fit_transform(Y) 
    return Y
