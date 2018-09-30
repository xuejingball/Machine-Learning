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
from data_preprocessing import getEcoliX, getEcoliY, getAdultX, getAdultY,getEcoliTestX, getEcoliTestY
import time

if __name__ == "__main__":
    acc = {}
    np.random.seed(0)
    ecoliX = getEcoliX()
    ecoliY = getEcoliY()
    ecoliTestX = getEcoliTestX()
    ecoliTestY = getEcoliTestY()
    
    adultX = getAdultX()
    adultX, adultTestX = adultX.iloc[:6000, :], adultX.iloc[6000:, :]
    adultY = getAdultY()
    adultY, adultTestY = adultY[:6000,], adultY[6000:,]
    
    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(4,), learning_rate='constant', learning_rate_init=0.2, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(ecoliX, ecoliY)
    print("ecoli", time.clock()-start_time, "seconds")
    acc['ecoli-train'] = accuracy_score(ecoliY, mlp.predict(ecoliX))
    acc['ecoli-test'] = accuracy_score(ecoliTestY, mlp.predict(ecoliTestX))

    start_time = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant', learning_rate_init=0.4, max_iter=500,early_stopping=True,random_state=5)
    mlp.fit(adultX, adultY)
    print("adult", time.clock()-start_time, "seconds")
    acc['adult-train'] = accuracy_score(adultY, mlp.predict(adultX))
    acc['adult-test'] = accuracy_score(adultTestY, mlp.predict(adultTestX))
    
    acc = pd.Series(acc)

    acc.to_csv('NN accuracy.csv')
    