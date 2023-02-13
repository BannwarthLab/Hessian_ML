from multiprocessing.sharedctypes import Value
from tkinter import Grid
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
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern ,RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from ml_functions_permutation import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import normalize
from sklearn.inspection import permutation_importance

#############################
#Init File Import for Information
#
 #+ glob.glob('tests/h2/H2_*/')# + glob.glob('tests/N2/N2_*/') + glob.glob('tests/O2/O2_*/') +glob.glob('tests/F2/F2_*/')
#mol_list = ['H2','N2','O2','F2','CO','HF']

list_test = []
list_true = []
r_score_diag_list = []
r_score_non_diag_list = []

molecules = ['H2','N2','O2','F2','CO','HF','H2O','NH3']
#molecules = [mol_list[molecule]]
X_df,y_df,col_name,N_at = gen_X_y_DF(molecules)

X_df = X_df.reset_index()
y_df = y_df.reset_index()

X_diag_prep, y_diag_prep, X_non_diag_prep, y_non_diag_prep = gen_feature_target_precursor(X_df,y_df)

grps_diag,grps_non_diag =  get_grps(X_diag_prep,X_non_diag_prep)

X_diag,y_diag =  transform_diag_prep(X_diag_prep,y_diag_prep,col_name)

X_non_diag,y_non_diag = transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep,col_name)

#############################
#Separation into Training and Test data
for rnd_state in [0,22,42,100,63]:#,54,96,35,408,74]:
    gss = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state = rnd_state)

    idx_diag = list(gss.split(X_diag, y_diag, grps_diag))

    idx_non_diag = list(gss.split(X_non_diag,y_non_diag,grps_non_diag))

    train_idx = idx_diag[0][0]
    test_idx =  idx_diag[0][1]

    X_train = X_diag[train_idx]
    X_test = X_diag[test_idx]
    grps_test = grps_diag[test_idx]

    y_train = y_diag[train_idx]
    y_test = y_diag[test_idx]

    train_idx_non_diag = idx_non_diag[0][0]
    test_idx_non_diag =  idx_non_diag[0][1]

    X_non_diag_train = X_non_diag[train_idx_non_diag]
    X_non_diag_test = X_non_diag[test_idx_non_diag]
    grps_test_non_diag = grps_non_diag[test_idx_non_diag]


    y_non_diag_train = y_non_diag[train_idx_non_diag]
    y_non_diag_test = y_non_diag[test_idx_non_diag]



    #############################
    # Training and Prediction via ML
    print('Start Fitting of Machine Learning')

    param_grid = [{'C' : [0.1,1,10,100],'gamma' : [100,10,1,0.1,0.01],'epsilon' : [10,1,0.1,0.001,0.0001], 
                    'kernel' : ['rbf']}]

    regr_diag =  RandomForestRegressor(n_estimators = 300,max_features=20,random_state=rnd_state,bootstrap=False) 

    #regr_diag = KNeighborsRegressor(n_neighbors= 25 ,weights='distance')
    #regr_diag = GridSearchCV(svm.SVR(),param_grid,cv = 2)

    regr_non_diag = RandomForestRegressor(n_estimators = 100,max_features=20,random_state=rnd_state,bootstrap=False)
    
    #regr_non_diag = GridSearchCV(svm.SVR(),param_grid,cv = 2)
    #regr_non_diag = KNeighborsRegressor(n_neighbors= 25 ,weights='distance')

    model_diag = regr_diag.fit(X_train, y_train)

    model_non_diag = regr_non_diag.fit(X_non_diag_train, y_non_diag_train)

    hess_diag_pred = regr_diag.predict(X_test)

    hess_non_diag_pred = regr_non_diag.predict(X_non_diag_test)

    #############################
    #Results and Plots of diagonal hessian matrix blocks
    rscore_diag = regr_diag.score(X_test,y_test)
    print('Score of diag model:',rscore_diag)
    r_score_diag_list.append(rscore_diag)

    #plot_diag_importances(regr_diag,col_name,rnd_state)

    plot_diag_perm_importances(model_diag,X_test,y_test,col_name,rnd_state)

    #############################
    #Training and Prediction via ML

    #############################
    #Results and Plots of prediction of non diagonal hessian matrix blocks
    rscore_non_diag = regr_non_diag.score(X_non_diag_test,y_non_diag_test)
    print('Score of non diag model:',rscore_non_diag)
    r_score_non_diag_list.append(rscore_non_diag)

    #plot_non_diag_importances(regr_non_diag,col_name,rnd_state)

    plot_non_diag_perm_importances(model_non_diag,X_non_diag_test,y_non_diag_test,col_name,rnd_state)

    #############################
    #Transformation of the hessian blocks into a full hessian to compute the frequencies
    #rnd = random.randint(0,len(test_idx)-1)

    for idx in range(len(np.unique(grps_test))):

        id_0 = np.unique(grps_test)[idx]
        
        temp = (X_diag_prep.loc[(X_diag_prep['mol_idx'] == id_0)])

        mol = int(temp['molecule'].iloc[0])
        file_num = int(temp['variation'].iloc[0])

        N_atoms = N_at.iloc[mol]
        
        N_diag = int(N_atoms)

        N_non_diag = int((N_atoms ** 2 - N_atoms)/2)

        id = id_0*N_diag

        ##########

        k = int(np.where(grps_test == id_0)[0][0])

        l = int(np.where(grps_test_non_diag == id_0)[0][0])
        
        num_hess_diag = N_diag * 9
        
        num_hess_non_diag = N_non_diag * 9

        ##########
        
        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
        inert_path_coord = f'{file_path[file_num]}'+'/inert_coord/coord.xyz'
        inert_path_hess = f'{file_path[file_num]}'+'/inert_coord/hessian'

        rot_arr = gen_rot_arr(file_path[file_num])

        hess = gen_full_hess_mat_from_vector(y_df,X_df,hess_diag_pred[k:k+num_hess_diag],hess_non_diag_pred[l:l+num_hess_non_diag],N_atoms,mol,file_num,rot_arr)

        coord_inert,head = import_coord(inert_path_coord)
        hess_inert =  np.array(pd.read_csv(inert_path_hess,delimiter = '\t',index_col = 0))
        #hess_inert = import_hess(inert_path_hess,coord_inert)

        hess_inert_proj,Q = project_hess(hess_inert,coord_inert)
        hess_inert_proj_mass = mass_weighted_hessian(hess_inert_proj.copy(),coord_inert['atoms'])
        freq_inert_proj_mass = freq(hess_inert_proj_mass)

        list_true.extend(freq_extract(freq_inert_proj_mass.copy()))

        hess_proj,Q= project_hess(hess,coord_inert)
        hess_proj_mass = mass_weighted_hessian(hess_proj.copy(),coord_inert['atoms'])
        freq_proj_mass = freq(hess_proj_mass)

        #print(freq_inert_proj_mass)
        #print(freq_proj_mass)

        list_test.extend(freq_extract(freq_proj_mass.copy()))


np.savetxt('plots/freq_ML',list_test)
np.savetxt('plots/freq_xTB',list_true)

np.savetxt('plots/rscore_diag',r_score_diag_list)
np.savetxt('plots/rscore_nondiag',r_score_non_diag_list)

