from msilib.schema import Feature
from pyparsing import col
from sklearn import preprocessing
from sklearn.cluster import k_means
from functions import *
from operator import matmul
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from scipy.spatial.transform import Rotation as rot_trafo
from math import log10 , floor
import glob as glob
import time as time
from scipy.stats import linregress 
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern ,RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from ml_functions import *

#############################
#Init File Import for Information
#
 #+ glob.glob('tests/h2/H2_*/')# + glob.glob('tests/N2/N2_*/') + glob.glob('tests/O2/O2_*/') +glob.glob('tests/F2/F2_*/')

molecules = ['H2','N2','O2','F2']
X_df = pd.DataFrame()
y_df = pd.DataFrame()
for mol in range(len(molecules)):
    file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
    X_df_mol = pd.DataFrame()
    y_df_mol = pd.DataFrame()
    for file in range(len(file_path)):
        temp,Hessian_Coll,col_name,y_df_temp = import_files(file_path[file])

        temp.insert(1, 'variation', file)
        temp.insert(1, 'molecule', mol)
        X_df_mol = pd.concat([X_df_mol,temp],axis = 0)
        y_df_mol = pd.concat([y_df_mol,y_df_temp],axis = 0)
    X_df = pd.concat([X_df,X_df_mol],axis = 0)
    y_df = pd.concat([y_df,y_df_mol],axis = 0)



X_df = X_df.reset_index(drop = True)
y_df = y_df.reset_index(drop = True)


###############################
#Generating features for diagonal hessian matrix blocks
N_diag = 62*2
N_non_diag = 62
X_diag_prep = np.empty([N_diag,40])
y_diag_prep = np.empty([N_diag,9])

X_non_diag_prep = np.empty([N_non_diag,200])
y_non_diag_prep = np.empty([N_non_diag,9])



k = 0
l = 0
for mol in range(len(molecules)):
    file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
    for var in range(len(file_path)):
        temp =  X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol)]
        apf_len = temp['apf'].max()
        for apf in range(apf_len+1):
            X_df_apf = X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (X_df['apf'] == apf)]
            if len(X_df_apf) == 1:
                X_diag_prep[k] = X_df_apf.iloc[0,-40:]
                y_diag_prep[k] = np.array(y_df.iloc[k+l]) 
                k+=1

            elif len(X_df_apf) == 2:

                X_non_diag_prep[l,:40] = X_df_apf.iloc[0,-40:]
                X_non_diag_prep[l,40:80] = X_df_apf.iloc[1,-40:]
                X_non_diag_prep[l,80:120] = (X_df_apf.iloc[0,-40:] + X_df_apf.iloc[1,-40:])/2
                X_non_diag_prep[l,120:160] = X_df_apf.iloc[0,-40:] * X_df_apf.iloc[1,-40:]
                X_non_diag_prep[l,160:200] = np.abs(X_df_apf.iloc[0,-40:] - X_df_apf.iloc[1,-40:])
                y_non_diag_prep[l,:] = np.array(y_df.iloc[l+k]) 
                l+=1

X_diag,y_diag,grps_diag =  transform_diag_prep(X_diag_prep,y_diag_prep)
X_non_diag,y_non_diag,grps_non_diag = transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep)


#############################
#Generate features for non diagonal Hessian matrix blocks

#X_non_diag,y_non_diag = non_diag_features(X_df,Hessian_Coll,apf_list,file_path)

#############################
#Separation into Training and Test data

gss = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state =2)
idx = list(gss.split(X_diag, y_diag, grps_diag))

train_idx = idx[0][0]
test_idx =  idx[0][1]

X_train = X_diag[train_idx]
X_test = X_diag[test_idx]

y_train = y_diag[train_idx]
y_test = y_diag[test_idx]




#############################
# Training and Prediction via ML
regr_diag = RandomForestRegressor( n_estimators = 100,random_state=42,bootstrap=False)
#regr_diag = GaussianProcessRegressor(kernel = Matern() ,random_state=42)

regr_diag.fit(X_train, y_train)

#############################
#Results and Plots of diagonal hessian matrix blocks

print('Score of diag model:',regr_diag.score(X_test,y_test))
importances = regr_diag.feature_importances_

ticks = np.array(range(len(X_diag[0])))

fig ,ax = plt.subplots()

ax.bar(ticks ,importances,alpha=1.0, width=0.15, color='orange')
ax.set_xlabel('Features')
ax.set_ylabel('R² coefficient')
ax.set_xticks(ticks )
ax.set_xticklabels(['x','y','z']+col_name,rotation=90)
plt.gcf().subplots_adjust(bottom=0.45)
fig.set_figwidth(15)
plt.savefig('diag_r2_coefficents.png')
plt.savefig('diag_r2_coefficents.svg')

#############################
#Separation into Training and Test data

idx = list(gss.split(X_non_diag, y_non_diag, grps_non_diag))

train_idx = idx[0][0]
test_idx =  idx[0][1]

X_non_diag_train = X_non_diag[train_idx]
X_non_diag_test = X_non_diag[test_idx]

y_non_diag_train = y_non_diag[train_idx]
y_non_diag_test = y_non_diag[test_idx]

#############################
#Training and Prediction via ML

regr_non_diag = RandomForestRegressor(n_estimators = 100,random_state=42, bootstrap=False)
#regr_non_diag = GaussianProcessRegressor(kernel = Matern(),random_state=42)

regr_non_diag.fit(X_non_diag_train, y_non_diag_train)

hess_non_diag_pred = regr_non_diag.predict(X_non_diag_test)
hess_diag_pred = regr_diag.predict(X_test)

#############################
#Results and Plots of prediction of non diagonal hessian matrix blocks
print('Score of non diag model:',regr_non_diag.score(X_non_diag_test,y_non_diag_test))

importances = regr_non_diag.feature_importances_

ticks = np.array(range(43))

fig ,ax = plt.subplots()

ax.bar(ticks[:3] ,importances[:3],alpha=1.0, width=0.15, color='black')
ax.bar(ticks[3:]-0.3 ,importances[3:43],alpha=1.0, width=0.15, label = 'atom A',color='green')
ax.bar(ticks[3:]-0.15 ,importances[43:83],alpha=1.0, width=0.15, label = 'atom B',color='orange')
ax.bar(ticks[3:] ,importances[83:123],alpha=1.0, width=0.15, label = 'arithmetic mean',color='blue')
ax.bar(ticks[3:]+0.15 ,importances[123:163],alpha=1.0, width=0.15, label = 'product',color='lime')
ax.bar(ticks[3:]+0.3 ,importances[163:203],alpha=1.0, width=0.15, label = 'absolute difference',color='red')

ax.legend()
ax.set_xlabel('Features')
ax.set_ylabel('R² coefficient')
ax.set_xticks(ticks)
ax.set_xticklabels(['x','y','z'] + col_name,rotation=90)
plt.gcf().subplots_adjust(bottom=0.45)
fig.set_figwidth(15)
plt.savefig('non_diag_r2_coefficents.png')
plt.savefig('non_diag_r2_coefficents.svg')


#############################
#Transformation of the hessian blocks into a full hessian to compute the frequencies
num = 0#18*7
file_num = 0#test_idx[num]//18

hess_diag = np.zeros([3,3])
hess_non_diag = np.zeros([3,3])

mol = 0
file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
init_path_coord = f'{file_path[file_num]}'+'/init_coord/coord.xyz'
R_00 = np.genfromtxt(f'{file_path[file_num]}'+'apf_coord/atoms_0_0HH/R_inert_apf.txt')
R_01 = np.genfromtxt(f'{file_path[file_num]}'+'apf_coord/atoms_0_1HH/R_inert_apf.txt')


k= int(num//9*9)
for i in range(3):
    for j in range(3):

        hess_diag[i,j] = hess_diag_pred[k]
        hess_non_diag[i,j] = hess_non_diag_pred[k]

        k += 1

hess = np.zeros([6,6])


hess[0:3,0:3] = matmul(matmul((R_00),hess_diag),np.transpose(R_00))
hess[3:6,3:6] = matmul(matmul((R_00),hess_diag),np.transpose(R_00))
hess[0:3,3:6] = matmul(matmul((R_01),hess_non_diag),np.transpose(R_01))
hess[3:6,0:3] = matmul(matmul((R_01),hess_non_diag),np.transpose(R_01))


#P = np.genfromtxt('h2/P_init_inert')
mol = 0
file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
init_path_coord = f'{file_path[file_num]}'+'/init_coord/coord.xyz'
coord,head = import_coord(init_path_coord)

hess_pred = hess #matmul(matmul(np.transpose(P),hess),P)
hessian_mass = mass_weighted_hessian(hess,coord['atoms'])
lamb, Q = linalg.eigh(hessian_mass)
freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))

print(freq1)
#hess_test = matmul(matmul(np.transpose(P),Hessian_Coll[5]),P)

file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
init_path_coord = f'{file_path[file_num]}'+'/init_coord/coord.xyz'


hess_diag_test = import_hess(f'{file_path[file_num]}'+'apf_coord/atoms_0_0HH/hessian',coord)[0:3,0:3]
hess_non_diag_test = import_hess(f'{file_path[file_num]}'+'apf_coord/atoms_0_1HH/hessian',coord)[0:3,3:6]

hess_test = np.zeros([6,6])

hess_test[0:3,0:3] = matmul(matmul((R_00),hess_diag_test),np.transpose(R_00))
hess_test[3:6,3:6] = matmul(matmul((R_00),hess_diag_test),np.transpose(R_00))
hess_test[0:3,3:6] = matmul(matmul((R_01),hess_non_diag_test),np.transpose(R_01))
hess_test[3:6,0:3] = matmul(matmul((R_01),hess_non_diag_test),np.transpose(R_01))

#hess_test = import_hess(f'{file_path[file_num]}'+'/init_coord/hessian',coord)
hessian_mass = mass_weighted_hessian(hess_test,coord['atoms'])
lamb, Q = linalg.eigh(hessian_mass)
freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))

print(freq1)


