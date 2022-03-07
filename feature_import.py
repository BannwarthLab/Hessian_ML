from multiprocessing.sharedctypes import Value
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
import random as random

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

molecules = ['N2'] #['H2','N2','O2','F2','CO','HF']

X_df,y_df,col_name,N_at = gen_X_y_DF(molecules)

X_df = X_df.reset_index(drop = True)

y_df = y_df.reset_index(drop = True)

X_diag_prep, y_diag_prep, X_non_diag_prep, y_non_diag_prep = gen_feature_target_precursor(X_df,y_df)

grps_diag,grps_non_diag =  get_grps(X_diag_prep,X_non_diag_prep)

X_diag,y_diag =  transform_diag_prep(X_diag_prep,y_diag_prep)

X_non_diag,y_non_diag = transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep)

#############################
#Separation into Training and Test data

gss = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state = 100)

idx_diag = list(gss.split(X_diag, y_diag, grps_diag))

idx_non_diag = list(gss.split(X_non_diag,y_non_diag,grps_non_diag))

train_idx = idx_diag[0][0]
test_idx =  idx_diag[0][1]

X_train = X_diag[train_idx]
X_test = X_diag[test_idx]

y_train = y_diag[train_idx]
y_test = y_diag[test_idx]

train_idx_non_diag = idx_non_diag[0][0]
test_idx_non_diag =  idx_non_diag[0][1]

X_non_diag_train = X_non_diag[train_idx_non_diag]
X_non_diag_test = X_non_diag[test_idx_non_diag]

y_non_diag_train = y_non_diag[train_idx_non_diag]
y_non_diag_test = y_non_diag[test_idx_non_diag]


#############################
# Training and Prediction via ML
print('Start Fitting of Machine Learning')
regr_diag = RandomForestRegressor(n_estimators = 100,random_state=0,bootstrap=False)
#regr_diag = GaussianProcessRegressor(kernel = Matern() ,random_state=42)

regr_non_diag = RandomForestRegressor(n_estimators = 100,random_state=0, bootstrap=False)
#regr_non_diag = GaussianProcessRegressor(kernel = Matern(),random_state=42)

regr_diag.fit(X_train, y_train)
regr_non_diag.fit(X_non_diag_train, y_non_diag_train)

hess_diag_pred = regr_diag.predict(X_test)
hess_non_diag_pred = regr_non_diag.predict(X_non_diag_test)

#############################
#Results and Plots of diagonal hessian matrix blocks

print('Score of diag model:',regr_diag.score(X_test,y_test))
importances = regr_diag.feature_importances_

ticks = np.array(range(len(X_diag[0])))

fig ,ax = plt.subplots()

ax.bar(ticks ,importances,alpha=1.0, width=0.15, color='orange')
ax.set_xlabel('Features')
ax.set_ylabel('R² coefficient')
ax.set_xticks(ticks)

ax.set_xticklabels(['x','y','z']+col_name,rotation=90)
plt.gcf().subplots_adjust(bottom=0.45)
fig.set_figwidth(15)

plt.savefig('diag_r2_coefficents.png')
plt.savefig('diag_r2_coefficents.svg')

np.array(range(int(max(grps_diag)+1)))

#############################
#Training and Prediction via ML


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
#rnd = random.randint(0,len(test_idx)-1)
rnd = 47
np.savetxt('grps',grps_diag)

id = grps_diag[test_idx[rnd]]*2

mol = int(X_diag_prep.loc[id,['molecule']])
file_num =int(X_diag_prep.loc[id,['variation']])

print('Molecule:',mol, 'and its variation:',file_num)

##########
def gen_full_hess_mat_from_vector(hess_diag_pred,hess_non_diag_pred):
    num_atoms = int(N_at.loc[mol,'Nat'])
    lenH = 3 * num_atoms
    clean_hess = np.zeros([lenH,lenH])
    idx_mol_var = X_df.loc[(X_df['molecule'] == mol) & (X_df['variation'] == file_num)].index.values.tolist()

    for i in idx_mol_var:
        A = int(y_df.loc[i,'atom1'])
        B = int(y_df.loc[i,'atom2'])
        if A == B:
            hess_vec = hess_diag_pred
        elif A != B:
            hess_vec = hess_non_diag_pred

        hess_mat = hess_vec_to_hess_block(hess_vec)
        clean_hess[3*A:3*A+3,3*B:3*B+3] = hess_mat 

    return clean_hess

k = 0
num_atoms = 2
num_hessians = num_atoms **2

hess_xy = gen_full_hess_mat_from_vector(hess_diag_pred[k:k+num_hessians],hess_non_diag_pred[k:k+num_hessians])

##########

hess_diag = np.zeros([3,3])
hess_non_diag = np.zeros([3,3])

file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
init_path_coord = f'{file_path[file_num]}'+'/init_coord/coord.xyz'

coord,head = import_coord(init_path_coord)

path_00 = glob.glob(f'{file_path[file_num]}'+'apf_coord/atoms_0_0*/')
path_01 = glob.glob(f'{file_path[file_num]}'+'apf_coord/atoms_0_1*/')

R_00 = np.genfromtxt(path_00[0]+'R_inert_apf.txt')
R_01 = np.genfromtxt(path_01[0]+'R_inert_apf.txt')

k= int((rnd+1)//18*18)
for i in range(3):
    for j in range(3):
        hess_diag[i,j] = hess_diag_pred[k]
        hess_non_diag[i,j] = hess_non_diag_pred[k]
        k += 1

hess = np.zeros([6,6])

hess[0:3,0:3] = matmul(matmul(np.transpose(R_00),hess_diag),(R_00))
hess[3:6,3:6] = matmul(matmul(np.transpose(R_00),hess_diag),(R_00))

hess[0:3,3:6] = matmul(matmul(np.transpose(R_01),hess_non_diag),(R_01))
hess[3:6,0:3] = matmul(matmul(np.transpose(R_01),hess_non_diag),(R_01))

np.savetxt('hess_pred.txt',hess)

hess_diag_test = import_hess(path_00[0]+'/hessian',coord)[0:3,0:3]
hess_non_diag_test = import_hess(path_01[0]+'/hessian',coord)[0:3,3:6]

hess_test = np.zeros([6,6])

hess_test[0:3,0:3] = matmul(matmul(np.transpose(R_00),hess_diag_test),(R_00))
hess_test[3:6,3:6] = matmul(matmul(np.transpose(R_00),hess_diag_test),(R_00))

hess_test[0:3,3:6] = matmul(matmul(np.transpose(R_01),hess_non_diag_test),(R_01))
hess_test[3:6,0:3] = matmul(matmul(np.transpose(R_01),hess_non_diag_test),(R_01))


hess_mass = mass_weighted_hessian(hess.copy(),coord['atoms'])
freq_mass = freq(hess_mass)

print(freq_mass)

hess_proj = project_hess(hess.copy())
hess_proj_mass = mass_weighted_hessian(hess_proj.copy(),coord['atoms'])
freq_mass_proj = freq(hess_proj_mass)

print(freq_mass_proj)

hess_test_mass = mass_weighted_hessian(hess_test.copy(),coord['atoms'])
freq_test_mass = freq(hess_test_mass)

print(freq_test_mass)

hess_test_proj = project_hess(hess_test.copy())
hess_test_proj_mass = mass_weighted_hessian(hess_test_proj.copy(),coord['atoms'])
freq_test_mass_proj = freq(hess_test_proj_mass)

print(freq_test_mass_proj)
