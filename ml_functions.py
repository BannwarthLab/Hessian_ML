from msilib.schema import Feature
from pyparsing import col
from sklearn import preprocessing
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
def non_diag_features(X_df,Hessian_Coll,apf_list,file_path):
    N_tot = len(X_df)
    X_non_diag = np.zeros([N_tot*9,203])
    y_non_diag = np.zeros(N_tot*9)
    k = 0
    for file in range(len(file_path)):
        for A in apf_list:
            for B in apf_list:
                if A != B:

                    Feature_Coll_A= np.array(X_df.loc[(X_df['file'] == file) & (X_df['index'] == A)])[:,2:]
                    Feature_Coll_B = np.array(X_df.loc[(X_df['file'] == file) & (X_df['index'] == B)])[:,2:]

                    for i in range(3):
                        for j in range(3):
                                #if A < B:
                                #print('i,j=',i,j)
                                #print('A,B=',A,B)
                                xyz = np.array([0,0,0])
                                xyz[i] += 1
                                xyz[j] += 1
                                
                                X_non_diag[k,0:3] = xyz
                                X_non_diag[k,3:43] = Feature_Coll_A
                                X_non_diag[k,43:83]= Feature_Coll_B
                                X_non_diag[k,83:123]= (Feature_Coll_A +Feature_Coll_B)/2
                                X_non_diag[k,123:163]= (Feature_Coll_A*Feature_Coll_B)
                                X_non_diag[k,163:203]= np.abs(Feature_Coll_A -Feature_Coll_B)

                                y_non_diag[k] = Hessian_Coll[file,i+3*A,j+3*B]
                                k+=1
    return X_non_diag,y_non_diag

def diag_features(X_df,Hessian_Coll,apf_list,file_path):
    N_tot = len(X_df)
    X = np.zeros([N_tot*9,43])
    y = np.zeros(N_tot*9)
    grps_diag = y.copy()

    k = 0
    l = 0
    for file in range(len(file_path)):
        for A in apf_list:
            Feature_Coll = np.array(X_df.loc[(X_df['file'] == file) & (X_df['index'] == A)])[:,2:]
            for i in range(3):
                for j in range(3):
                    xyz = np.array([0,0,0])
                    grps_diag[k] = int(l)
                    xyz[i] += 1
                    xyz[j] += 1
                    X[k,0:3] = xyz
                    X[k,3:]  = Feature_Coll
                    y[k] = Hessian_Coll[file,i+3*A,j+3*A]
                    k+=1
            l+=1
    return X,y,grps_diag

def import_files(file_path_mol):
    #############################
    #Import Files
    file_path = glob.glob(os.path.join(file_path_mol,'apf_coord/atoms_*'))

    feature_path = file_path.copy()
    coord_path = file_path.copy()
    info_path = file_path.copy()
    hess_path = file_path.copy()
    for file in range(len(file_path)):
        feature_path[file] = os.path.join(file_path[file],'ml_feature.csv')
        coord_path[file] = os.path.join(file_path[file],'coord.xyz')
        hess_path[file] = os.path.join(file_path[file], 'hessian')
        info_path[file] = os.path.join(file_path[file], 'atoms.txt')

    #ml_feature = pd.concat(pd.concat([pd.DataFrame(data = {'file':[1]}),pd.read_csv(f)],axis =1 ) for f in feature_path)
    ml_feature = pd.DataFrame()
    hessian_df = pd.DataFrame()
    for f in range(len(feature_path)):
        A,B = np.genfromtxt(info_path[f], delimiter =',')
        
        if A == B:
            temp = pd.read_csv(feature_path[f]).iloc[[A]]
        elif A != B:  
            temp = pd.read_csv(feature_path[f]).iloc[[A,B]]

        temp.insert(0,'apf',f)
        ml_feature = pd.concat([ml_feature,temp])

        coord,head = import_coord(coord_path[f])
        hessian = import_hess(hess_path[f],coord)

        k = 0
        temp_hesssian_df = pd.DataFrame(columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'],index = [f])
        for i in range(3):
            for j in range(3):
                temp_hesssian_df.iloc[0,k] =  hessian[3*int(A)+i,3*int(B)+j]
                k+=1

        hessian_df = pd.concat([hessian_df,temp_hesssian_df],axis =0)

    ml_feature = ml_feature.reset_index()
    N_tot = len(ml_feature)

    #apf list for which atoms are on the z axis


    #ml_feature generate hessian
    Hessian_Coll = np.empty([len(file_path),6,6])
    for file in range(len(file_path)):
        coord,head = import_coord(coord_path[file])
        hessian = import_hess(hess_path[file],coord)
        Hessian_Coll[file] = hessian

    ##############################
    #Extract features
    file = ml_feature.loc[:,['index','apf']]

    CN = ml_feature.loc[:,('coordination number','delta coordination number')]
    charges = ml_feature.loc[:,('atomic partial charges','delta partial charges')]

    dipm = ml_feature.loc[:,('dipm_atom_x','dipm_atom_y','dipm_atom_z',
                            'dipm_delta_x','dipm_delta_y','dipm_delta_z',
                            'delta dipm only mull x','delta dipm only mull y','delta dipm only mull z')]

    qm = ml_feature.loc[:,('delta qm only Z xx','delta qm only Z yy',' delta qm only Z zz',
                        'delta qm only mull xx',' delta qm only mull yy',' delta qm only mull zz')]

    energy_based = ml_feature.loc[:,('response (a.u.)','gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext')]

    ##############################
    #Extract features which need to be transformed

    qm_atom = ml_feature.loc[:,('qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_zx','qm_delta_yz')]
    qm_delta = ml_feature.loc[:,('qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_zx','qm_delta_yz')]

    #Transform to vector by QM x I
    qm_atom_mat = np.zeros([len(qm_atom),3])
    qm_delta_mat = np.zeros([len(qm_atom),3])

    I = np.array([1,1,1])

    for k in range(len(qm_atom.iloc[:,0])):
        qm_atom_mat[k,:] = matmul(qm_matrix(qm_atom.iloc[k,:]),I)
        qm_delta_mat[k,:] = matmul(qm_matrix(qm_delta.iloc[k,:]),I)


    qm_atom_mat_df = pd.DataFrame(qm_atom_mat,columns = ['qm_x','qm_y','qm_z'])
    qm_delta_mat_df = pd.DataFrame(qm_delta_mat,columns = ['qm_delta_x','qm_delta_y','qm_delta_z'])

    col_name = (CN.columns.values.tolist() + dipm.columns.values.tolist() + qm_atom_mat_df.columns.values.tolist()+ qm_delta_mat_df.columns.values.tolist() + qm.columns.values.tolist() + energy_based.columns.values.tolist())

    #Generate the Feature DataFrame
    X_df = pd.concat([file,CN,dipm,qm_atom_mat_df,qm_delta_mat_df,qm,energy_based],axis =1)

    return X_df,Hessian_Coll,col_name,hessian_df