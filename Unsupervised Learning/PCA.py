# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from data_preprocessing import getEcoliX, getEcoliY, getAdultX, getAdultY,getEcoliTestX, getEcoliTestY
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from helpers import nn_arch_ecoli, nn_reg_ecoli, nn_arch_adult, nn_reg_adult
from sklearn.tree import DecisionTreeClassifier

out = './PCA/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)
ecoliX = getEcoliX()
ecoliY = getEcoliY()
ecoliTestX = getEcoliTestX()
ecoliTestY = getEcoliTestY()

adultX = getAdultX()
adultX, adultTestX = adultX.iloc[:6000, :], adultX.iloc[6000:, :]
adultY = getAdultY()
adultY, adultTestY = adultY[:6000,], adultY[6000:,]

dims1 = [1,2,3,4,5,6,7]
dims2 = [1,2,3,4,5,6,7,8,9,10]
#raise
#%% data for 1
svm = SVC(kernel = "linear", random_state = 0, C = 6)
pca = PCA(random_state=5)
eigenvalue = {}
acc = {}
for dim in dims1:
    pca.set_params(n_components=dim)
    tmp = pca.fit_transform(ecoliX)
    svm.fit(tmp, ecoliY)
    testX = pca.fit_transform(ecoliTestX)
    acc[dim] = accuracy_score(ecoliTestY, svm.predict(testX))
    
eigenvalue = pd.Series(data=eigenvalue) 
eigenvalue.to_csv(out+'ecoli scree.csv')
acc = pd.Series(acc)
acc.to_csv(out+'ecoli svm validate.csv')

dt = DecisionTreeClassifier(random_state=0)
pca = PCA(random_state=5)
eigenvalue = {}
acc = {}
for dim in dims2:
    pca.set_params(n_components=dim)
    pca.fit(adultX)
    eigenvalue[dim] = pca.explained_variance_ratio_
    X_train = pca.transform(adultX)
    X_test = pca.transform(adultTestX)
    dt.fit(X_train, adultY)
    score = dt.score(X_test, adultTestY)
    acc[dim] = score
    
eigenvalue = pd.Series(eigenvalue) 
eigenvalue.to_csv(out+'adult scree.csv')
acc = pd.Series(acc)
acc.to_csv(out+'adult svm validate.csv')

raise
#%% Data for 2

grid ={'pca__n_components':dims1,'NN__alpha':nn_reg_ecoli,'NN__hidden_layer_sizes':nn_arch_ecoli}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(ecoliX,ecoliY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'ecoli dim red.csv')


grid ={'pca__n_components':dims2,'NN__alpha':nn_reg_adult,'NN__hidden_layer_sizes':nn_arch_adult}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(adultX,adultY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'adult dim red.csv')
raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 4
pca = PCA(n_components=dim,random_state=5)

ecoliX2 = pca.fit_transform(ecoliX)
ecoli2 = pd.DataFrame(np.hstack((ecoliX2,np.atleast_2d(ecoliY).T)))
cols = list(range(ecoli2.shape[1]))
cols[-1] = 'Class'
ecoli2.columns = cols
ecoli2.to_csv(out+'ecoli_dr.csv')
#ecoli2.to_hdf(out+'datasets.hdf','ecoli',complib='blosc',complevel=9)

dim = 3
pca = PCA(n_components=dim,random_state=5)
adultX2 = pca.fit_transform(adultX)
adult2 = pd.DataFrame(np.hstack((adultX2,np.atleast_2d(adultY).T)))
cols = list(range(adult2.shape[1]))
cols[-1] = 'Class'
adult2.columns = cols
adult2.to_csv(out+'adult_dr.csv')
#adult2.to_hdf(out+'datasets.hdf','adult',complib='blosc',complevel=9)