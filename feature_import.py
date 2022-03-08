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
mol_list = ['H2','N2','O2','F2','CO','HF']

list_test = []
list_true = []
for molecule in mol_list:
#molecules = ['N2'] #['H2','N2','O2','F2','CO','HF']
    molecules = [molecule]
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

    plot_diag_importances(regr_diag,col_name)

    #############################
    #Training and Prediction via ML


    #############################
    #Results and Plots of prediction of non diagonal hessian matrix blocks
    print('Score of non diag model:',regr_non_diag.score(X_non_diag_test,y_non_diag_test))

    plot_non_diag_importances(regr_non_diag,col_name)

    #############################
    #Transformation of the hessian blocks into a full hessian to compute the frequencies
    #rnd = random.randint(0,len(test_idx)-1)
    rnd_list = []
    for m in range(int(len(test_idx-1)//18)):
        rnd_list.append(int(m*18))

    for rnd in rnd_list:

        id = grps_diag[test_idx[rnd]]*2
        mol = int(X_diag_prep.loc[id,['molecule']])
        file_num =int(X_diag_prep.loc[id,['variation']])

        print('Molecule:',mol, 'and its variation:',file_num)

        ##########

        k = (rnd+1)//18*18
        num_atoms = 2
        num_hessians = num_atoms ** 2 * 9

        ##########
        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
        init_path_coord = f'{file_path[file_num]}'+'/init_coord/coord.xyz'
        coord,head = import_coord(init_path_coord)
        rot_arr = gen_rot_arr(file_path[file_num])

        hess = gen_full_hess_mat_from_vector(y_df,X_df,hess_diag_pred[k:k+num_hessians],hess_non_diag_pred[k:k+num_hessians],N_at,mol,file_num,rot_arr)

        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
        inert_path_coord = f'{file_path[file_num]}'+'/inert_coord/coord.xyz'
        inert_path_hess = f'{file_path[file_num]}'+'/inert_coord/hessian'

        coord_inert,head = import_coord(inert_path_coord)
        hess_inert =  np.array(pd.read_csv(inert_path_hess,delimiter = '\t',index_col = 0))
        #hess_inert = import_hess(inert_path_hess,coord_inert)

        hess_inert_proj,Q = project_hess(hess_inert.copy(),coord_inert)
        hess_inert_proj_mass = mass_weighted_hessian(hess_inert_proj.copy(),coord_inert['atoms'])
        freq_inert_proj_mass = freq(hess_inert_proj_mass)

        list_true.append(np.amax(freq_inert_proj_mass))

        hess_proj,Q = project_hess(hess.copy(),coord)
        hess_proj_mass = mass_weighted_hessian(hess_proj.copy(),coord['atoms'])
        freq_proj_mass = freq(hess_proj_mass)

        list_test.append(np.amax(freq_proj_mass))


np.savetxt('freq_ML',list_test)
np.savetxt('freq_xTB',list_true)