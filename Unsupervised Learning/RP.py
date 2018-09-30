

#%% Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg_ecoli,nn_arch_ecoli, nn_reg_adult,nn_arch_adult
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from data_preprocessing import getEcoliX, getEcoliY, getAdultX, getAdultY,getEcoliTestX, getEcoliTestY, getAdultTestX, getAdultTestY
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from itertools import product


out = './RP/'
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

dims1 = range(1,8)
dims2 = [2,4,6,8,10,20,30,40,50,60,70,80]
#raise
#%% data for 1
tmp = defaultdict(dict)
svm = SVC(kernel = "linear", random_state = 0, C = 6)
sp = SparseRandomProjection()
acc = defaultdict(dict)
for i, dim in product(range(10),dims1):
    sp.set_params(random_state=i, n_components=dim)
    X_train = sp.fit_transform(ecoliX)
    tmp[dim][i] = pairwiseDistCorr(sp.fit_transform(ecoliX), ecoliX)
    svm.fit(X_train, ecoliY)
    X_test = sp.transform(ecoliTestX)
    acc[dim][i] = accuracy_score(ecoliTestY, svm.predict(X_test))

tmp =pd.DataFrame(tmp)
tmp.to_csv(out+'ecoli scree1.csv')
acc = pd.DataFrame(acc)
acc.to_csv(out + 'ecoli svm validate.csv')


tmp = defaultdict(dict)
dt = DecisionTreeClassifier(random_state=0)
sp = SparseRandomProjection()
acc = defaultdict(dict)
for i, dim in product(range(10),dims2):
    sp.set_params(random_state=i, n_components=dim)
    X_train = sp.fit_transform(adultX)
    tmp[dim][i] = pairwiseDistCorr(sp.fit_transform(adultX), adultX)
    dt.fit(X_train, adultY)
    X_test = sp.transform(adultTestX)
    acc[dim][i] = accuracy_score(adultTestY, dt.predict(X_test))

tmp =pd.DataFrame(tmp)
tmp.to_csv(out+'adult scree1.csv')
acc = pd.DataFrame(acc)
acc.to_csv(out + 'adult dt validate.csv')

raise
#%% Data for 2

grid ={'rp__n_components':dims1,'NN__alpha':nn_reg_ecoli,'NN__hidden_layer_sizes':nn_arch_ecoli}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(ecoliX,ecoliY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'ecoli dim red.csv')


grid ={'rp__n_components':dims2,'NN__alpha':nn_reg_adult,'NN__hidden_layer_sizes':nn_arch_adult}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(adultX,adultY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'adult dim red.csv')
raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 5
rp = SparseRandomProjection(n_components=dim,random_state=5)

ecoliX2 = rp.fit_transform(ecoliX)
ecoli2 = pd.DataFrame(np.hstack((ecoliX2,np.atleast_2d(ecoliY).T)))
cols = list(range(ecoli2.shape[1]))
cols[-1] = 'Class'
ecoli2.columns = cols
#ecoli2.to_hdf(out+'datasets.hdf','ecoli',complib='blosc',complevel=9)
ecoli2.to_csv(out+'ecoli_dr.csv')

dim = 10
rp = SparseRandomProjection(n_components=dim,random_state=5)
adultX2 = rp.fit_transform(adultX)
adult2 = pd.DataFrame(np.hstack((adultX2,np.atleast_2d(adultY).T)))
cols = list(range(adult2.shape[1]))
cols[-1] = 'Class'
adult2.columns = cols
#adult2.to_hdf(out+'datasets.hdf','adult',complib='blosc',complevel=9)
adult2.to_csv(out+'adult_dr.csv')