# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:47:00 2018

@author: Jing
"""

from sklearn import  datasets, metrics
from clustertesters import adult_ExpectationMaximizationTestCluster as emtc
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


if __name__ == "__main__":
    acc = {}
    ecoli = pd.read_csv("adult_dr.csv", index_col=0)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(4,), learning_rate='constant', learning_rate_init=0.2, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("adult", time.clock()-start_time, "seconds")
    acc['adult-train'] = accuracy_score(y, mlp.predict(X))
    acc['adult-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    
    ecoli = pd.read_csv("ecoli_dr.csv", index_col=0)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant', learning_rate_init=0.4, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("ecoli", time.clock()-start_time, "seconds")
    acc['ecoli-train'] = accuracy_score(y, mlp.predict(X))
    acc['ecoli-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    acc = pd.Series(acc)
    print(acc)
    acc.to_csv('NN accuracy.csv')



