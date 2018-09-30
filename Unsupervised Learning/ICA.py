

#%% Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from helpers import nn_arch_ecoli, nn_reg_ecoli, nn_arch_adult, nn_reg_adult
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from data_preprocessing import getEcoliX, getEcoliY, getAdultX, getAdultY,getEcoliTestX, getEcoliTestY
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

out = './ICA/'

np.random.seed(0)
ecoliX = getEcoliX()
ecoliY = getEcoliY()
ecoliTestX = getEcoliTestX()
ecoliTestY = getEcoliTestY()

adultX = getAdultX()
adultY = getAdultY()
adultX = getAdultX()
adultX, adultTestX = adultX.iloc[:6000, :], adultX.iloc[6000:, :]
adultY = getAdultY()
adultY, adultTestY = adultY[:6000,], adultY[6000:,]

dims1 = range(1,8)
dims2 = range(1,16)
#raise
#%% data for 1
svm = SVC(kernel = "linear", random_state = 0, C = 6)
ica = FastICA(random_state=5)
kurt = {}
acc = {}
for dim in dims1:
    ica.set_params(n_components=dim, max_iter=500, tol=0.1)
    tmp = ica.fit_transform(ecoliX)
    svm.fit(tmp, ecoliY)
    testX = ica.transform(ecoliTestX)
    acc[dim] = accuracy_score(ecoliTestY, svm.predict(testX))
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()


kurt = pd.Series(kurt) 
kurt.to_csv(out+'ecoli scree.csv')
acc = pd.Series(acc)
acc.to_csv(out + 'ecoli svm validate.csv')
tmp.to_csv(out + 'ecoli kurtosis.csv')

dt = DecisionTreeClassifier(random_state=0)
ica = FastICA(random_state=5)
kurt = {}
acc = {}
for dim in dims2:
    ica.set_params(n_components=dim, max_iter=500, tol=0.1)
    tmp = ica.fit_transform(adultX)
    dt.fit(tmp, adultY)
    testX = ica.fit_transform(adultTestX)
    acc[dim] = accuracy_score(adultTestY, dt.predict(testX))
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'adult scree.csv')
acc = pd.Series(acc)
acc.to_csv(out + 'adult svm validate.csv')
tmp.to_csv(out + 'adult kurtosis.csv')
raise

#%% Data for 2

grid ={'ica__n_components':dims1,'NN__alpha':nn_reg_ecoli,'NN__hidden_layer_sizes':nn_arch_ecoli}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(ecoliX,ecoliY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'ecoli dim red.csv')


grid ={'ica__n_components':dims2,'NN__alpha':nn_reg_adult,'NN__hidden_layer_sizes':nn_arch_adult}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(adultX,adultY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'adult dim red.csv')
raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 2
ica = FastICA(n_components=dim,random_state=5)

ecoliX2 = ica.fit_transform(ecoliX)
kur = pd.DataFrame(ecoliX2).kurt(axis=0)
print(kur)
ecoli2 = pd.DataFrame(np.hstack((ecoliX2,np.atleast_2d(ecoliY).T)))
cols = list(range(ecoli2.shape[1]))
cols[-1] = 'Class'
ecoli2.columns = cols
#ecoli2.to_hdf(out+'datasets.hdf','ecoli',complib='blosc',complevel=9)
ecoli2.to_csv(out+'ecoli_dr.csv')

dim = 4
ica = FastICA(n_components=dim,random_state=5)
adultX2 = ica.fit_transform(adultX)
kur = pd.DataFrame(adultX2).kurt(axis=0)
print(kur)
adult2 = pd.DataFrame(np.hstack((adultX2[:,:3],np.atleast_2d(adultY).T)))
cols = list(range(adult2.shape[1]))
cols[-1] = 'Class'
adult2.columns = cols
#adult2.to_hdf(out+'datasets.hdf','adult',complib='blosc',complevel=9)
adult2.to_csv(out+'adult_dr.csv')