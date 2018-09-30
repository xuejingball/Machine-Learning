# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:21:51 2018

@author: Jing
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from collections import defaultdict

if __name__ == "__main__":
    acc = defaultdict(dict)
    ecoli = pd.read_csv("adult_cluster_kmeans.csv", index_col=0)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(4,), learning_rate='constant', learning_rate_init=0.2, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("adult kmeans", time.clock()-start_time, "seconds")
    acc['kmeans']['adult-train'] = accuracy_score(y, mlp.predict(X))
    acc['kmeans']['adult-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    ecoli = pd.read_csv("adult_cluster_em.csv", index_col=0)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(4,), learning_rate='constant', learning_rate_init=0.2, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("adult em", time.clock()-start_time, "seconds")
    acc['em']['adult-train'] = accuracy_score(y, mlp.predict(X))
    acc['em']['adult-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    
    ecoli = pd.read_csv("ecoli_cluster_kmeans.csv", index_col=None)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant', learning_rate_init=0.4, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("ecoli kmeans", time.clock()-start_time, "seconds")
    acc['kmeans']['ecoli-train'] = accuracy_score(y, mlp.predict(X))
    acc['kmeans']['ecoli-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    ecoli = pd.read_csv("ecoli_cluster_em.csv", index_col=None)
    X = ecoli.iloc[:, :-1]
    y = ecoli.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant', learning_rate_init=0.4, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(X_train, y_train)
    print("ecoli em", time.clock()-start_time, "seconds")
    acc['em']['ecoli-train'] = accuracy_score(y, mlp.predict(X))
    acc['em']['ecoli-test'] = accuracy_score(y_test, mlp.predict(X_test))
    
    acc = pd.Series(acc)
    print(acc)
    acc.to_csv('NN cluster accuracy.csv')