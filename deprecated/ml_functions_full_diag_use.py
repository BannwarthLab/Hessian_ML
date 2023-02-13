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

def project_hess(hess_v,coord):
    idx = find_trans_rot(hess_v,coord)
    lamb,Q = linalg.eigh(hess_v)
    hess_projected_v = hess_v.copy()

    for i in idx:
        i = int(i)
        hess_projected_v -= lamb[i] * np.outer(Q.T[i], Q.T[i])

    return hess_projected_v,Q

def find_trans_rot(hess,coord):

    Nat = len(coord)

    overlap_mat = np.zeros([6,3*Nat])
    
    trans_x = np.array([1.,0.,0.])
    trans_y = np.array([0.,1.,0.])
    trans_z = np.array([0.,0.,1.])

    for i in range(Nat):
        overlap_mat[0,3*i:3*i+3] = trans_x
        overlap_mat[1,3*i:3*i+3] = trans_y
        overlap_mat[2,3*i:3*i+3] = trans_z

        overlap_mat[3,3*i:3*i+3] = np.array([0.,coord.loc[i,'z'],-coord.loc[i,'y']])
        overlap_mat[4,3*i:3*i+3] = np.array([-coord.loc[i,'z'],0.,coord.loc[i,'x']])
        overlap_mat[5,3*i:3*i+3] = np.array([coord.loc[i,'y'],-coord.loc[i,'x'],0.])

    overlap_mat = overlap_mat / Nat
    overlap_mat = 1/(linalg.norm(overlap_mat,1)) * overlap_mat


    lamb, Q = linalg.eigh(hess)

    M = matmul(overlap_mat,Q)

    
    norm_x = np.array(linalg.norm(coord.loc[:,'x']))
    
    norm_y = np.array(linalg.norm(coord.loc[:,'y']))

    norm_z = np.array(linalg.norm(coord.loc[:,'z']))

    idx_len = 6 

    if (norm_x + norm_y) < 1e-6 or (norm_y + norm_z) < 1e-6 or (norm_z + norm_x)< 1e-6:
        idx_len = 5

    M_sum = np.zeros(3*Nat)
    for i in range(len(M_sum)):
        M_sum[i] = np.sum(np.abs(M[:,i]))

    idx_list = np.zeros(idx_len)
    for i in range(idx_len):
        idx = np.where(M_sum == np.amax(M_sum))[0][0]
        idx_list[i] = int(idx)
        M_sum[idx] -= M_sum[idx]

    return idx_list

def freq(hess_v):
    lamb, Q = linalg.eigh(hess_v)
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
    return freq_val

def freq_extract(freq):
    list_freq = []
    while np.amax(freq) > 1 :
        idx = np.where(freq == np.amax(freq))[0][0]
        list_freq.append(freq[idx])
        freq[idx] = 0

    return list_freq

def gen_X_y_DF(molecules):
    X_df = pd.DataFrame()
    y_df = pd.DataFrame()
    Nat = pd.DataFrame()

    for mol in range(len(molecules)):
        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')
        Nat.loc[mol,'Nat'] = len(import_coord(f'{file_path[0]}/init_coord/coord.xyz')[0])
        X_df_mol = pd.DataFrame()
        y_df_mol = pd.DataFrame()
        
        for file in range(len(file_path)):
            temp,col_name,y_df_temp = import_files(file_path[file])


            temp.insert(1, 'variation', file)
            temp.insert(1, 'molecule', mol)
            X_df_mol = pd.concat([X_df_mol,temp],axis = 0)
            y_df_mol = pd.concat([y_df_mol,y_df_temp],axis = 0)

        X_df = pd.concat([X_df,X_df_mol],axis = 0)
        y_df = pd.concat([y_df,y_df_mol],axis = 0)
    return X_df,y_df,col_name,Nat



def gen_feature_target_precursor(X_df,y_df):
    col_name = X_df.columns.values.tolist()

    col_name_info = col_name[:-40]

    col_name_temp = col_name[-40:]

    col_name_A = [s + '_A' for s in col_name][-40:]

    col_name_B = [s + '_B' for s in col_name][-40:]

    col_name_Arith = [s + '_Arith' for s in col_name][-40:]

    col_name_Prod = [s + '_Prod' for s in col_name][-40:]

    col_name_AbsDiff = [s + '_AbsDiff' for s in col_name][-40:]

    col_name_y = y_df.columns.values.tolist()
    
    X_diag_prep = pd.DataFrame()
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
            X_temp_diag = X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (X_df['block'] == 'diag')].reset_index(drop =True)
            y_temp_diag = y_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (y_df['block'] == 'diag')].reset_index(drop =True)

            X_temp_non_diag = X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (X_df['block'] == 'nondiag')].reset_index(drop =True)
            y_temp_non_diag = y_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol) & (y_df['block'] == 'nondiag')].reset_index(drop =True)

            for i in range(len(y_temp_diag)):
                y_diag_prep.loc[l+i,col_name_y] = y_temp_diag.loc[i,col_name_y]
                X_diag_prep.loc[l+i,col_name] = X_temp_diag.loc[i,col_name]

            l += len(y_temp_diag)

            for i in range(0,len(X_temp_non_diag),2):

                j =i + 1
                y_non_diag_prep.loc[k+i,col_name_y] = y_temp_non_diag.loc[i,col_name_y]

                X_non_diag_prep.loc[k+i,col_name_info] = X_temp_non_diag.loc[i,col_name_info]

                X_non_diag_prep.loc[k+i,col_name_A] = np.array(X_temp_non_diag.loc[i,col_name_temp])
                X_non_diag_prep.loc[k+i,col_name_B] = np.array(X_temp_non_diag.loc[j,col_name_temp])

                X_non_diag_prep.loc[k+i,col_name_Arith] = np.array(X_temp_non_diag.loc[i,col_name_temp] + X_temp_non_diag.loc[j,col_name_temp])
                X_non_diag_prep.loc[k+i,col_name_Prod] = np.array(X_temp_non_diag.loc[i,col_name_temp] * X_temp_non_diag.loc[j,col_name_temp])

                X_non_diag_prep.loc[k+i,col_name_AbsDiff] = np.array(np.abs(X_temp_non_diag.loc[i,col_name_temp] - X_temp_non_diag.loc[j,col_name_temp]))

                i,j = j,i

                y_non_diag_prep.loc[k+i,col_name_y] = y_temp_non_diag.loc[i,col_name_y]

                X_non_diag_prep.loc[k+i,col_name_info] = X_temp_non_diag.loc[i,col_name_info]

                X_non_diag_prep.loc[k+i,col_name_A] = np.array(X_temp_non_diag.loc[i,col_name_temp])
                X_non_diag_prep.loc[k+i,col_name_B] = np.array(X_temp_non_diag.loc[j,col_name_temp])

                X_non_diag_prep.loc[k+i,col_name_Arith] = np.array(X_temp_non_diag.loc[i,col_name_temp] + X_temp_non_diag.loc[j,col_name_temp])
                X_non_diag_prep.loc[k+i,col_name_Prod] = np.array(X_temp_non_diag.loc[i,col_name_temp] * X_temp_non_diag.loc[j,col_name_temp])

                X_non_diag_prep.loc[k+i,col_name_AbsDiff] = np.array(np.abs(X_temp_non_diag.loc[i,col_name_temp] - X_temp_non_diag.loc[j,col_name_temp]))
                
            k += 2*len(X_temp_non_diag)


            """
            temp =  X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol)]
            apf_len = temp['apf'].max()
            for apf in range(apf_len+1):
                X_df_apf = X_df.loc[(X_df['variation'] == var) & (X_df['molecule'] == mol)& (X_df['apf'] == apf)]

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
            """
    
    X_diag_prep = X_diag_prep.reset_index(drop = True).drop(columns = ['block'])

    y_diag_prep = y_diag_prep.reset_index(drop = True).drop(columns = ['block'])

    X_non_diag_prep = X_non_diag_prep.reset_index(drop = True).drop(columns = ['block'])

    y_non_diag_prep = y_non_diag_prep.reset_index(drop = True).drop(columns = ['block'])

    return  X_diag_prep, y_diag_prep, X_non_diag_prep, y_non_diag_prep

def gen_full_hess_mat_from_vector(y_df,X_df,hess_diag_pred,hess_non_diag_pred,num_atoms,mol,file_num,rot_arr):

    lenH = 3 * int(num_atoms)
    clean_hess = np.zeros([lenH,lenH])
    idx_mol_var = X_df.loc[(X_df['molecule'] == mol) & (X_df['variation'] == file_num)].index.values.tolist()

    print(idx_mol_var)

    mat_list = np.arange(6)#gen_sym_mat_list(int(num_atoms))

    m = 0
    o = 0
    for l in range(len(idx_mol_var)):

        i = idx_mol_var[l]

        
        A = int(y_df.loc[i,'atom1'])
        B = int(y_df.loc[i,'atom2'])

        k = mat_list[m]

        if A == B:
            rot = rot_arr[k]
            hess_vec = hess_diag_pred[9*A:9*A+9]
            m+=1
            hess_mat = hess_vec_to_hess_block(hess_vec)
            clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            
        elif A != B:

            hess_vec = hess_non_diag_pred[o:o+9]
            o += 9

            rot = rot_arr[k]

            if B > A:
                m += 1
                hess_mat = hess_vec_to_hess_block(hess_vec)
                clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            elif A < B:
                m+= 0
                hess_mat = hess_vec_to_hess_block(hess_vec)
                clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))
   

        



    return clean_hess

def get_grps(X_diag_prep,X_non_diag_prep):

    grps_non_diag = np.zeros(len(X_non_diag_prep)*9)

    grps_diag = np.zeros(len(X_diag_prep)*9)

    molecules = int(X_diag_prep['molecule'].max() +1)

    l = 0
    for mol in range(molecules):
        temp =  X_diag_prep.loc[(X_diag_prep['molecule'] == mol)]
        var_len = int(temp['variation'].max() + 1)

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
    
def gen_rot_arr(path_variation):
    path_apf_list = glob.glob(f'{path_variation}'+'apf_coord/atoms_*')
    print(path_apf_list)
    rot_arr = []
    for path_apf in path_apf_list:
        rot = np.genfromtxt(os.path.join(path_apf,'R_inert_apf.txt'))
        rot_arr.append(rot)
    return rot_arr

def gen_sym_mat_list(N):
    idx_list = []
    k = N
    l = 0
    m = 1
    while k > 0:
        for _ in range(k):
            idx_list.append(l)
            l+= 1 
        o = N -1

        if k >1:

            for _ in range(N-k+1):
                idx_list.append(m)
                m += o
                o-= 1
        m = N-k+2
        k -= 1
    return idx_list

def hess_block_to_hess_vec(hess_mat,A,B):
    k = 0
    hess_vec = np.zeros(9)
    for i in range(3):
        for j in range(3):
            hess_vec[k] =  hess_mat[3*int(A)+i,3*int(B)+j]
            k+=1
    return hess_vec

def hess_vec_to_hess_block(hess_vec):
    k = 0 
    hess_mat = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            hess_mat[i,j] = hess_vec[k]
            k += 1
    return hess_mat

            

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

    k = 0
    for f in range(len(feature_path)):

        A,B = np.genfromtxt(info_path[f], delimiter =',')

        feature_df = pd.read_csv(feature_path[f])

        coord,head = import_coord(coord_path[f])
        hessian = import_hess(hess_path[f],coord)

        col_hess=['xx','xy','xz','yx','yy','yz','zx','zy','zz']

        hess_temp = hess_block_to_hess_vec(hessian,A,B)

        if A == B:
            hessian_df.loc[k,'block'] = 'diag'
            hessian_df.loc[k,'atom1'] = A
            hessian_df.loc[k,'atom2'] = B

            hessian_df.loc[k,col_hess] = hess_temp
            temp = feature_df.iloc[[A]]

            temp.insert(0,'atom',[A])
            temp.insert(0,'block','diag')


        elif A != B:  

            hessian_df.loc[k,'block'] = 'nondiag'
            hessian_df.loc[k,'atom1'] = A
            hessian_df.loc[k,'atom2'] = B
            hessian_df.loc[k,col_hess] = hess_temp
      
            temp = feature_df.iloc[[A,B]]
            temp.insert(0,'atom',[A,B])
            temp.insert(0,'block','nondiag')

            hess_temp = hess_block_to_hess_vec(hessian,B,A)

            k+=1
            hessian_df.loc[k,'block'] = 'nondiag'
            hessian_df.loc[k,'atom1'] = B
            hessian_df.loc[k,'atom2'] = A

            hessian_df.loc[k,col_hess] = hess_temp

        temp.insert(0,'apf',f)
        ml_feature = pd.concat([ml_feature,temp])
        k += 1

    ml_feature = ml_feature.reset_index()
    #apf list for which atoms are on the z axis
    
    #ml_feature generate hessian
    for file in range(len(file_path)):
        coord,head = import_coord(coord_path[file])
        hessian = import_hess(hess_path[file],coord)

    ##############################
    #Extract features
    file = ml_feature.loc[:,['index','apf','block','atom']]

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

    return X_df,col_name,hessian_df



def transform_diag_prep(X_diag_prep,y_diag_prep):
    col_name_X = X_diag_prep.columns.values.tolist()[-40:]
    col_name_y = y_diag_prep.columns.values.tolist()[-9:]

    len_df = len(X_diag_prep)*9

    X_diag = np.zeros([len_df,43])
    y_diag = np.zeros(len_df)

    l = 0
    for A in range(len(X_diag_prep)):
        m = 0
        for i in range(3):
            for j in range(3):
                xyz = np.array([0,0,0])
                xyz[i] += 1
                xyz[j] += 1

                X_diag[l,0:3] = xyz

                X_diag[l,3:]  = X_diag_prep.loc[A,col_name_X]
                y_diag[l] = y_diag_prep.loc[A,[col_name_y[m]]]

                l += 1
                m += 1

    return X_diag,y_diag

def transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep):

    col_name_X = X_non_diag_prep.columns.values.tolist()[-200:]
    col_name_y = y_non_diag_prep.columns.values.tolist()[-9:]

    len_df = len(X_non_diag_prep)*9
    X_non_diag = np.zeros([len_df,203])
    y_non_diag = np.zeros(len_df)


    l = 0
    for A in range(len(X_non_diag_prep)):
        m = 0
        for i in range(3):
            for j in range(3):
                xyz = np.array([0,0,0])
                xyz[i] += 1
                xyz[j] += 1
                X_non_diag[l,0:3] = xyz
                X_non_diag[l,3:]  = X_non_diag_prep.loc[A,col_name_X]
                y_non_diag[l] = y_non_diag_prep.loc[A,[col_name_y[m]]]
                l += 1
                m += 1

    return X_non_diag,y_non_diag

def plot_non_diag_importances(regr_non_diag,col_name):
    importances = regr_non_diag.feature_importances_

    ticks = np.array(range(43))
    fig ,ax = plt.subplots()
    ax.bar(ticks[:3] ,importances[:3],alpha=1.0, width=0.15, color='black')
    ax.bar(ticks[3:]-0.3 ,importances[3:43],alpha=1.0, width=0.15, label = 'atom A')
    ax.bar(ticks[3:]-0.15 ,importances[43:83],alpha=1.0, width=0.15, label = 'atom B')
    ax.bar(ticks[3:] ,importances[83:123],alpha=1.0, width=0.15, label = 'arithmetic mean')
    ax.bar(ticks[3:]+0.15 ,importances[123:163],alpha=1.0, width=0.15, label = 'product')
    ax.bar(ticks[3:]+0.3 ,importances[163:203],alpha=1.0, width=0.15, label = 'absolute difference')

    ax.legend()
    ax.set_xlabel('Features')
    ax.set_ylabel('R² coefficient')
    ax.set_xticks(ticks)
    ax.set_xticklabels(['x','y','z'] + col_name,rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    fig.set_figwidth(15)

    plt.savefig('non_diag_r2_coefficents.png')
    plt.savefig('non_diag_r2_coefficents.svg')

    return


def plot_diag_importances(regr_diag,col_name):
    importances = regr_diag.feature_importances_

    ticks = np.array(range(len(importances)))

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

    return