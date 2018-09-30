

#%% Imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from helpers import   nn_arch_ecoli,nn_reg_ecoli,nn_arch_adult,nn_reg_adult,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from data_preprocessing import getEcoliX, getEcoliY, getAdultX, getAdultY,getEcoliTestX, getEcoliTestY
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


out = './RF/'

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


dims1 = [1,2,3,4,5,6,7]
dims2 = range(1,16)
    #%% data for 1
    
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
fs_ecoli = rfc.fit(ecoliX,ecoliY).feature_importances_ 
fs_adult = rfc.fit(adultX,adultY).feature_importances_ 

tmp = pd.Series(np.sort(fs_ecoli)[::-1])
tmp.to_csv(out+'ecoli scree.csv')

tmp = pd.Series(np.sort(fs_adult)[::-1])
tmp.to_csv(out+'adult scree.csv')



    #%% Data for 2

#svm = SVC(kernel = "linear", random_state = 0, C = 6)
#acc = {}
#for dim in dims1:
#    filtr = ImportanceSelect(rfc,dim)
#    X_train = filtr.fit_transform(ecoliX,ecoliY)
#    svm.fit(X_train, ecoliY)
#    X_test = filtr.transform(ecoliTestX)
#    acc[dim] = accuracy_score(ecoliTestY, svm.predict(X_test))
#
#acc = pd.Series(acc)
#acc.to_csv(out + 'ecoli svm validate.csv')
fs_ecoli = rfc.fit(ecoliX,ecoliY)
model = SelectFromModel(fs_ecoli, prefit=True)
ecoliX_new = model.transform(ecoliX)
svm = SVC(kernel = "linear", random_state = 0, C = 6)
svm.fit(ecoliX_new, ecoliY)
ecoliTestX_new = model.transform(ecoliTestX)
acc = accuracy_score(ecoliTestY, svm.predict(ecoliTestX_new))
print(ecoliX_new.shape, acc)
    
fs_adult = rfc.fit(adultX,adultY)
model = SelectFromModel(fs_adult, prefit=True)
adultX_new = model.transform(adultX)
dt = DecisionTreeClassifier(random_state=0)
dt.fit(adultX_new, adultY)
adultTestX_new = model.transform(adultTestX)
acc = accuracy_score(adultTestY, dt.predict(adultTestX_new))
print(adultX_new.shape, acc)

raise
    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
fs_ecoli = rfc.fit(ecoliX,ecoliY)
model = SelectFromModel(fs_ecoli, prefit=True)
ecoliX2 = model.transform(ecoliX)
ecoli2 = pd.DataFrame(np.hstack((ecoliX2,np.atleast_2d(ecoliY).T)))
cols = list(range(ecoli2.shape[1]))
cols[-1] = 'Class'
ecoli2.columns = cols
ecoli2.to_csv(out+'ecoli_dr.csv')
#ecoli2.to_hdf(out+'datasets.hdf','ecoli',complib='blosc',complevel=9)

# selectfrommodel autoselect 15 attributes from adult dataset
fs_adult = rfc.fit(adultX,adultY)
model = SelectFromModel(fs_adult, prefit=True)
adultX2 = model.transform(adultX)
adult2 = pd.DataFrame(np.hstack((adultX2,np.atleast_2d(adultY).T)))
cols = list(range(adult2.shape[1]))
cols[-1] = 'Class'
adult2.columns = cols
adult2.to_csv(out+'adult_dr.csv')
#adult2.to_hdf(out+'datasets.hdf','adult',complib='blosc',complevel=9)
