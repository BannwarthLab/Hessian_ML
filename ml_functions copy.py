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



def gen_X_y_DF(molecules):
    X_df = pd.DataFrame()
    y_df = pd.DataFrame()
    N_diag = 0
    N_non_diag = 0
    for mol in range(len(molecules)):
        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
        X_df_mol = pd.DataFrame()
        y_df_mol = pd.DataFrame()
        for file in range(len(file_path)):
            temp,Hessian_Coll,col_name,y_df_temp,N_diag_temp,N_non_diag_temp = import_files(file_path[file])
            print(y_df_temp)
            print(temp)
            N_diag += N_diag_temp
            N_non_diag += N_non_diag_temp

            temp.insert(1, 'variation', file)
            temp.insert(1, 'molecule', mol)
            X_df_mol = pd.concat([X_df_mol,temp],axis = 0)
            y_df_mol = pd.concat([y_df_mol,y_df_temp],axis = 0)
        X_df = pd.concat([X_df,X_df_mol],axis = 0)
        y_df = pd.concat([y_df,y_df_mol],axis = 0)
    return X_df,y_df,col_name,N_diag,N_non_diag



def gen_feature_target_precursor(X_df,y_df):
    col_name = X_df.columns.values.tolist()

    col_name_A = [s + '_A' for s in col_name]

    col_name_B = [s + '_B' for s in col_name]

    col_name_Arith = [s + '_Arith' for s in col_name][-40:]

    col_name_Prod = [s + '_Prod' for s in col_name][-40:]

    col_name_AbsDiff = [s + '_AbsDiff' for s in col_name][-40:]
    
    X_diag_prep = pd.DataFrame(columns = col_name)
    y_diag_prep = pd.DataFrame()

    #X_non_diag_prep = np.empty([N_non_diag,200])
    X_non_diag_prep = pd.DataFrame()

    y_non_diag_prep = pd.DataFrame()

    k = 0
    l = 0
    for mol in range(X_df['molecule'].max()+1):
        temp =  X_df.loc[(X_df['molecule'] == mol)]
        var_len = temp['variation'].max() + 1
        for var in range(var_len):
            temp =  X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol)]
            apf_len = temp['apf'].max()
            for apf in range(apf_len+1):
                X_df_apf = X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (X_df['apf'] == apf)]

                if len(X_df_apf) == 1:
                    X_diag_prep = pd.concat([X_diag_prep,X_df_apf.iloc[[0]]],axis = 0)
                    y_diag_prep = pd.concat([y_diag_prep,y_df.iloc[[k+l]]],axis = 0)
                    k+=1

                elif len(X_df_apf) == 2:

                    X_non_diag_prep.loc[l,col_name[:-40]] = np.array(X_df_apf.iloc[0,:-40])

                    X_non_diag_prep.loc[l,col_name_A[-40:]] = np.array(X_df_apf.iloc[0,-40:])
                    X_non_diag_prep.loc[l,col_name_B[-40:]] = np.array(X_df_apf.iloc[1,-40:])

                    X_non_diag_prep.loc[l,col_name_Arith] =np.array((X_df_apf.iloc[0,-40:] + X_df_apf.iloc[1,-40:])/2)
                    X_non_diag_prep.loc[l,col_name_Prod] = np.array(X_df_apf.iloc[0,-40:] * X_df_apf.iloc[1,-40:])
                    X_non_diag_prep.loc[l,col_name_AbsDiff] = np.array(np.abs(X_df_apf.iloc[0,-40:] - X_df_apf.iloc[1,-40:]))
                    
                    y_non_diag_prep =  pd.concat([y_non_diag_prep,y_df.iloc[[k+l]]],axis = 0)

                    l+=1


    X_diag_prep = X_diag_prep.reset_index()
    y_diag_prep = y_diag_prep.reset_index()

    X_non_diag_prep = X_non_diag_prep.reset_index()
    y_non_diag_prep = y_non_diag_prep.reset_index()

    return  X_diag_prep, y_diag_prep, X_non_diag_prep, y_non_diag_prep


def get_grps(X_diag_prep,X_non_diag_prep):
    grps_non_diag = np.zeros(len(X_non_diag_prep)*9)
    grps_diag = np.zeros(len(X_diag_prep)*9)
    molecules = X_diag_prep['molecule'].max() +1
    l = 0
    for mol in range(molecules):
        temp =  X_diag_prep.loc[(X_diag_prep['molecule'] == mol)]
        var_len = temp['variation'].max() + 1 

        for var in range(var_len):

            strucs_diag = X_diag_prep.loc[(X_diag_prep['molecule'] == mol) & (X_diag_prep['variation'] == var)].index.tolist()
            for j in strucs_diag:
                for i in range(9):
                    grps_diag[9*j+i] = l

            strucs_non_diag = X_non_diag_prep.loc[(X_non_diag_prep['molecule'] == mol) & (X_non_diag_prep['variation'] == var)].index.tolist()
            for j in strucs_non_diag:
                for i in range(9):
                    grps_non_diag[9*j+i] = l

            l+=1
    return grps_diag,grps_non_diag

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
    N_diag = 0
    N_non_diag = 0
    for f in range(len(feature_path)):
        A,B = np.genfromtxt(info_path[f], delimiter =',')
        feature_df = pd.read_csv(feature_path[f])
        
        if A == B:
            temp = feature_df.iloc[[A]]
            N_diag += 1

        elif A != B:  
            temp = feature_df.iloc[[A,B]]
            N_non_diag += 1

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

    return X_df,Hessian_Coll,col_name,hessian_df,N_diag,N_non_diag



def transform_diag_prep(X_diag_prep,y_diag_prep):

    len_df = len(X_diag_prep)*9
    X_diag = np.zeros([len_df,43])
    y_diag = np.zeros(len_df)

    grps_diag = y_diag.copy()
    l = 0
    for A in range(len(X_diag_prep)):
        m = 1
        for i in range(3):
            for j in range(3):
                xyz = np.array([0,0,0])
                xyz[i] += 1
                xyz[j] += 1
                X_diag[l,0:3] = xyz
                X_diag[l,3:]  = X_diag_prep.iloc[A,-40:]
                y_diag[l] = y_diag_prep.iloc[A,m]
                grps_diag[l] = A//2

                l += 1
                m += 1
    return X_diag,y_diag

def transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep):
    len_df = len(X_non_diag_prep)*9
    X_non_diag = np.zeros([len_df,203])
    y_non_diag = np.zeros(len_df)

    grps_non_diag = y_non_diag.copy()

    l = 0
    for A in range(len(X_non_diag_prep)):
        m = 1
        for i in range(3):
            for j in range(3):
                xyz = np.array([0,0,0])
                xyz[i] += 1
                xyz[j] += 1
                X_non_diag[l,0:3] = xyz
                X_non_diag[l,3:]  = X_non_diag_prep.iloc[A,-200:]
                y_non_diag[l] = y_non_diag_prep.iloc[A,m]
                grps_non_diag[l] = int(str(1) + str(int(X_non_diag_prep.loc[A,'molecule']))+ str(int(X_non_diag_prep.loc[A,'variation'])))
                l += 1
                m += 1
    return X_non_diag,y_non_diag