from sklearn import preprocessing
from sklearn.inspection import permutation_importance
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
import time as time

modIP = 'perm'

def project_hess(hess_v,coord):
    idx = find_trans_rot(hess_v.copy(),coord.copy())
    lamb,Q = linalg.eigh(hess_v)
    hess_projected_v = hess_v.copy()

    for i in idx:
        i = int(i)
        hess_projected_v -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)

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

def wavenumber(lamb):
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
    return freq_val

def frequency(lamb):
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi))
    return freq_val

def force_constant(lamb,atoms):
    m_sum = 0
    for i in range(len(atoms)):
        m_sum += 1/elements_dict[atoms[i]]
    mu = 1/m_sum
    fc = mu*lamb
    return fc


def freq_extract(freq):
    freq = freq.copy()
    list_freq = []
    while np.amax(freq) > 1e-3 :
        idx = np.where(freq == np.amax(freq))[0][0]
        list_freq.append(freq[idx])
        freq[idx] = 0

    return list(list_freq)

def get_feature_name(feature_col_name):

    #feature_col_name_A = [s + '_A' for s in feature_col_name]
    #feature_col_name_B = [s + '_B' for s in feature_col_name]
    feature_name_Arith = [s + '_Arith' for s in feature_col_name]
    feature_name_Prod = [s + '_Prod' for s in feature_col_name]
    feature_name_AbsDiff = [s + '_AbsDiff' for s in feature_col_name]
    feature_name_full = ['atom1','atom2','y_idx','y','pos','Rab']#'nucA','nucB',

    for i in [feature_name_Arith,feature_name_Prod,feature_name_AbsDiff]:# [feature_col_name_A,feature_col_name_B,
        feature_name_full.extend(i)

    return feature_name_full

def gen_X_y_DF(molecules):
    Nat = pd.DataFrame()

    k = 0 
    temp_all = []

    for mol in range(len(molecules)):

        print(f'Importing Molecule No. {mol}')

        feature_file_path = glob.glob(f'tests/{molecules[0]}/{molecules[0]}_*/')
        feature_path = os.path.join(feature_file_path[0],'init_coord/ml_feature.csv')
        feature_df = pd.read_csv(feature_path)
        print(feature_file_path[0])
        col = extract_feature(feature_df.iloc[0],'yx')

        feature_name_full = get_feature_name(col.index.values.tolist())
        feature_name_full.insert(0, 'block')
        feature_name_full.insert(0, 'variation')
        feature_name_full.insert(0, 'molecule')
        feature_name_full.insert(0, 'mol_idx')

        file_path = glob.glob(f'tests/{molecules[mol]}/{molecules[mol]}_*/')

        Nat.loc[mol,'Nat'] = len(import_coord(f'{file_path[0]}/init_coord/coord.xyz')[0])

        print('Generating DataFrame')

        for file in range(len(file_path)):
            temp = import_files(file_path[file],feature_name_full)
        
            for i in range(len(temp)):
                temp[i].insert(0,  file)
                temp[i].insert(0, mol)
                temp[i].insert(0, k)            

            temp_all.extend(temp)
            k += 1


        #temp_all = list(np.array(temp_all).T)

        X_df = pd.DataFrame(temp_all,columns=feature_name_full)

    return X_df,feature_name_full,Nat


def gen_full_hess_mat_from_vector(y_df,hess_diag_pred,hess_non_diag_pred,num_atoms,mol,file_num,rot_arr):

    lenH = 3 * int(num_atoms)
    clean_hess = np.zeros([lenH,lenH])
    idx_mol_var = y_df.loc[(y_df['molecule'] == mol) & (y_df['variation'] == file_num) & (y_df['y_idx'] == 'xx')].index.values.tolist()
    mat_list = np.arange(int(num_atoms+(num_atoms**2 - num_atoms)/2))#gen_sym_mat_list(int(num_atoms))

    o = 0
    for l in range(len(idx_mol_var)):

        i = idx_mol_var[l]

        A = int(y_df.loc[i,'atom1'])
        B = int(y_df.loc[i,'atom2'])

        k = mat_list[l]
        if A == B:
            rot = rot_arr[k]
            hess_vec = hess_diag_pred[9*A:9*A+9]

            hess_mat = hess_vec_to_hess_block(hess_vec)
            clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            
        elif A != B:
            hess_vec = hess_non_diag_pred[o:o+9]
            o += 9
            rot = rot_arr[k]
            hess_mat = hess_vec_to_hess_block(hess_vec)

            clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            hess_mat = hess_vec_to_hess_block(hess_vec)
            clean_hess[3*B:3*B+3,3*A:3*A+3] = np.transpose(matmul(matmul(np.transpose(rot),hess_mat),(rot)))


    return clean_hess

def get_grps(X_diag_prep,X_non_diag_prep):

    grps_diag = np.array(X_diag_prep['mol_idx'])
    grps_non_diag = np.array(X_non_diag_prep['mol_idx'])

    return grps_diag,grps_non_diag
    
def gen_rot_arr(path_variation):
    path_apf_list = glob.glob(f'{path_variation}'+'apf_coord/atoms_*')
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

def extract_feature(ml_feature,y_idx):
    CN = ml_feature.loc[['coordination number','delta coordination number']]

    dipm_atom = ml_feature.loc[['dipm_atom_x','dipm_atom_y','dipm_atom_z']]
    dipm_delta = ml_feature.loc[['dipm_delta_x','dipm_delta_y','dipm_delta_z']]
    dipm_only_mull = ml_feature.loc[['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']]

    # qm_delta_only_Z = ml_feature.loc[['delta qm only Z xx','delta qm only Z yy',' delta qm only Z zz']]

    # qm_delta_only_Z = qm_delta_only_Z.rename(index= {' delta qm only Z zz':'delta qm only Z zz'})

    # qm_delta_only_mull = ml_feature.loc[['delta qm only mull xx',' delta qm only mull yy',' delta qm only mull zz']]

    # qm_delta_only_mull = qm_delta_only_mull.rename(index= {' delta qm only mull yy' : 'delta qm only mull yy',' delta qm only mull zz' : 'delta qm only mull zz' })

    qm_atom = ml_feature.loc[['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_xz','qm_atom_yz']]
    qm_delta = ml_feature.loc[['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_xz','qm_delta_yz']]

    energy_based = ml_feature.loc[['response (a.u.)','gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']]  # 

    ##############################
    #Extract features which need to be transformed

    if y_idx == 'yx':

        perm_a = ['x','y','z']
        perm_aa = ['xx','yy','zz','xy','xz','yz']
        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    elif y_idx == 'xy':

        perm_a = ['y','x','z']
        perm_aa = ['yy','xx','zz','xy','yz','xz']

        sign_aa = [1,1,1,1,-1,-1]
        sign_a = [1,1,-1]

    elif y_idx == 'xz':

        perm_a = ['z','x','y']
        perm_aa = ['zz','xx','yy','xz','yz','xy']

        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    
    elif y_idx == 'yz':

        perm_a = ['z','y','x']
        perm_aa = ['zz','yy','xx','yz','xz','xy']

        sign_aa = [1,1,1,1,-1,-1]
        sign_a = [1,1,-1]


    elif y_idx == 'zx':

        perm_a = ['x','z','y']
        perm_aa = ['xx','zz','yy','xz','xy','yz']

        sign_aa = [1,1,1,1,-1,-1]
        sign_a = [1,1,-1]

    elif y_idx == 'zy':

        perm_a = ['y','z','x']
        perm_aa = ['yy','zz','xx','yz','xy','xz']

        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    elif y_idx == 'xx':

        perm_a = ['x','y','z']
        perm_aa = ['xx','yy','zz','xy','xz','yz']

        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    elif y_idx == 'yy':

        perm_a = ['y','z','x']
        perm_aa = ['yy','zz','xx','yz','xy','xz']

        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    elif y_idx == 'zz':

        perm_a = ['z','x','y']
        perm_aa = ['zz','xx','yy','xz','yz','xy']

        sign_aa = [1,1,1,1,1,1]
        sign_a = [1,1,1]

    else:
        print('Not in list')

    dipm_atom[['dipm_atom_x','dipm_atom_y','dipm_atom_z']] = np.array([sign_a[0]*dipm_atom[f'dipm_atom_{perm_a[0]}'], sign_a[1]*dipm_atom[f'dipm_atom_{perm_a[1]}'],sign_a[2]*dipm_atom[f'dipm_atom_{perm_a[2]}']])

    dipm_delta[['dipm_delta_x','dipm_delta_y','dipm_delta_z']] = np.array([sign_a[0]*dipm_delta[f'dipm_delta_{perm_a[0]}'],sign_a[1]*dipm_delta[f'dipm_delta_{perm_a[1]}'],sign_a[2]*dipm_delta[f'dipm_delta_{perm_a[2]}']])

    dipm_only_mull[['delta dipm only mull x','delta dipm only mull x','delta dipm only mull z']] = np.array([sign_a[0]*dipm_only_mull[f'delta dipm only mull {perm_a[0]}'],sign_a[1]*dipm_only_mull[f'delta dipm only mull {perm_a[1]}'],sign_a[2]*dipm_only_mull[f'delta dipm only mull {perm_a[2]}']])

    #qm_delta_only_Z[['delta qm only Z xx','delta qm only Z yy','delta qm only Z zz']] =  np.array([sign_a[0]*qm_delta_only_Z[f'delta qm only Z {perm_a[0]}{perm_a[0]}'],sign_a[1]*qm_delta_only_Z[f'delta qm only Z {perm_a[1]}{perm_a[1]}'],sign_a[2]*qm_delta_only_Z[f'delta qm only Z {perm_a[2]}{perm_a[2]}']])

    #qm_delta_only_mull[['delta qm only mull xx','delta qm only mull yy','delta qm only mull zz']] = np.array([sign_a[0]*qm_delta_only_mull[f'delta qm only mull {perm_a[0]}{perm_a[0]}'],sign_a[1]*qm_delta_only_mull[f'delta qm only mull {perm_a[1]}{perm_a[1]}'],sign_a[2]*qm_delta_only_mull[f'delta qm only mull {perm_a[2]}{perm_a[2]}']])
    
    qm_atom[['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_xz','qm_atom_yz']] =   np.array([sign_aa[0]*qm_atom[f'qm_atom_{perm_aa[0]}'],sign_aa[1]*qm_atom[f'qm_atom_{perm_aa[1]}'],sign_aa[2]*qm_atom[f'qm_atom_{perm_aa[2]}'],
                                                                                                        sign_aa[3]*qm_atom[f'qm_atom_{perm_aa[3]}'],sign_aa[4]*qm_atom[f'qm_atom_{perm_aa[4]}'],sign_aa[5]*qm_atom[f'qm_atom_{perm_aa[5]}']])

    qm_delta[['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_xz','qm_delta_yz']] =   np.array([sign_aa[0]*qm_delta[f'qm_delta_{perm_aa[0]}'],sign_aa[1]*qm_delta[f'qm_delta_{perm_aa[1]}'],sign_aa[2]*qm_delta[f'qm_delta_{perm_aa[2]}'],
                                                                                                                sign_aa[3]*qm_delta[f'qm_delta_{perm_aa[3]}'],sign_aa[4]*qm_delta[f'qm_delta_{perm_aa[4]}'],sign_aa[5]*qm_delta[f'qm_delta_{perm_aa[5]}']])

    #Transform to vector by QM x I
    qm_atom_mat = np.zeros(3)
    qm_delta_mat = np.zeros(3)
 
    I = np.array([1,1,1])

    qm_atom_mat = matmul(qm_matrix(qm_atom,'qm_atom_'),I)
    qm_delta_mat = matmul(qm_matrix(qm_delta,'qm_delta_'),I)

    qm_atom_mat_df = pd.Series(qm_atom_mat, index = ['qm_atom_x','qm_atom_y','qm_atom_z'])
    qm_delta_mat_df = pd.Series(qm_delta_mat,index = ['qm_delta_x','qm_delta_y','qm_delta_z'])

    #col_name = (CN.columns.values.tolist() + dipm.columns.values.tolist() + qm_atom_mat_df.columns.values.tolist()+ qm_delta_mat_df.columns.values.tolist() + qm.columns.values.tolist())

    #Generate the Feature DataFrame
    X_df = pd.concat([CN,dipm_atom,dipm_delta,dipm_only_mull,qm_atom_mat_df,qm_delta_mat_df,energy_based],axis =0)#,qm_delta_only_Z,qm_delta_only_mull,energy_based],axis =0)
    #X_df = pd.concat([file,CN,dipm,qm_atom_mat_df,qm_delta_mat_df,qm],axis =1)
    return X_df

def import_files(file_path_mol,feature_name_full):
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


    X_all = []
    k = 0
    for f in range(len(feature_path)):

        A,B = np.genfromtxt(info_path[f], delimiter =',').astype('int')

        feature_df = pd.read_csv(feature_path[f])

        coord,head = import_coord(coord_path[f])

        atomA = coord.iloc[A,0]
        atomB = coord.iloc[B,0]

        nucA = ELEMENTS2Z[atomA]
        nucB = ELEMENTS2Z[atomB]

        hessian = import_hess(hess_path[f],coord)

        col_hess=['xx','xy','xz','yx','yy','yz','zx','zy','zz']

        list_hess = np.array([0,1,2,
                              1,0,2,
                              2,2,3])

        hess_temp = hess_block_to_hess_vec(hessian,A,B)

        for i in range(len(col_hess)):
            X = []
            if A == B:
                X.extend(['diag'])
            elif A != B:
                X.extend(['nondiag'])
            
            X.extend([A])
            X.extend([B])

            X.extend([col_hess[i]])
            X.extend([hess_temp[i]])

            X.extend([list_hess[i]])

            #X.extend([nucA])
            #X.extend([nucB])

            X.extend([linalg.norm(coord.iloc[int(A),1:] - coord.iloc[int(B),1:])])

            X_A = np.array(extract_feature(feature_df.iloc[int(A)],col_hess[i]))
            X_B = np.array(extract_feature(feature_df.iloc[int(B)],col_hess[i]))

            if col_hess[i] in ['yx','zx','zy']:
                
                #X.extend(X_B)
                #X.extend(X_A)
                X.extend( (X_A + X_B )/ 2)
                X.extend(X_A * X_B)
                X.extend(np.abs(X_A - X_B))

            else:

                #X.extend(X_A)
                #X.extend(X_B)
                X.extend((X_A + X_B )/ 2)
                X.extend(X_A * X_B)
                X.extend(np.abs(X_A - X_B))
               
            X_all.append(X)

        k += 9
    
    return X_all

def transform_diag_prep(X_diag_prep,y_diag_prep,col_name):
    N_Features = len(col_name)

    col_name_X = X_diag_prep.columns.values.tolist()[-N_Features:]
    col_name_y = y_diag_prep.columns.values.tolist()[-9:]

    len_df = len(X_diag_prep)*9

    X_diag = np.zeros([len_df,N_Features+3])
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

def transform_non_diag_prep(X_non_diag_prep,y_non_diag_prep,col_name):
    N_Features = len(col_name)*3
    col_name_X = X_non_diag_prep.columns.values.tolist()[-N_Features:]
    col_name_y = y_non_diag_prep.columns.values.tolist()[-9:]

    len_df = len(X_non_diag_prep)*9
    X_non_diag = np.zeros([len_df,N_Features+3])
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

def plot_non_diag_importances(regr_non_diag,col_name,info):

    N_features = len(col_name)
    importances = regr_non_diag.feature_importances_

    np.savetxt(f'plots/importances_non_diag_{info}',importances)

    ticks = np.array(range(3+N_features))
    fig ,ax = plt.subplots()
    ax.bar(ticks[:3]-0.15 ,importances[:3],alpha=1.0, width=0.15, color='black')
    ax.bar(ticks[3:] ,importances[3:N_features+3],alpha=1.0, width=0.15, label = 'arithmetic mean')#'atom A')
    ax.bar(ticks[3:]+0.15 ,importances[N_features+3:2*N_features+3],alpha=1.0, width=0.15, label = 'product')#'atom B')
    ax.bar(ticks[3:] ,importances[2*N_features+3:3*N_features+3],alpha=1.0, width=0.15, label = 'absolute difference')#'arithmetic mean')
    #ax.bar(ticks[3:]+0.15 ,importances[123:163],alpha=1.0, width=0.15, label = 'product')
    #ax.bar(ticks[3:]+0.3 ,importances[163:203],alpha=1.0, width=0.15, label = 'absolute difference')

    ax.legend()
    ax.set_xlabel('Features')
    ax.set_ylabel('R² coefficient')
    ax.set_xticks(ticks)
    ax.set_xticklabels(['x','y','z'] + col_name,rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    fig.set_figwidth(15)

    plt.savefig(f'plots/non_diag_r2_coefficents_{info}.png')
    plt.savefig(f'plots/non_diag_r2_coefficents__{info}.svg')

    return

def plot_non_diag_perm_importances(model,X,y,col_name,info):

    N_features = 40
    N= 2
    importances = permutation_importance(model,X,y,n_repeats=15).importances_mean
    np.savetxt(f'plots/importances_non_diag_{info}',importances)

    ticks = np.array(range(N_features+N))
    fig ,ax = plt.subplots()
    ax.bar(ticks[0] ,importances[0],alpha=1.0, width=0.15,color = 'blue')
    ax.bar(ticks[1] ,importances[1],alpha=1.0, width=0.15,color = 'blue')

    ax.bar(ticks[N:N_features+N]-0.3 ,importances[N:N_features+N],alpha=1.0, width=0.15, label = 'A')
    ax.bar(ticks[N:N_features+N]-0.15,importances[N_features+N:2*N_features+N],alpha=1.0, width=0.15, label = 'B')#'atom A')
    ax.bar(ticks[N:N_features+N]     , importances[2*N_features+N:3*N_features+N],alpha=1.0, width=0.15, label = 'arithmetic mean')#'atom B')
    ax.bar(ticks[N:N_features+N]+0.15,importances[3*N_features+N:4*N_features+N],alpha=1.0, width=0.15, label = 'product')#'arithmetic mean')
    ax.bar(ticks[N:N_features+N]+0.3 ,importances[4*N_features+N:5*N_features+N],alpha=1.0, width=0.15, label = 'absolute difference')

    #ax.bar(ticks[167:207]+0.3 ,importances[163:203],alpha=1.0, width=0.15, label = 'absolute difference')

    ax.legend()
    ax.set_xlabel('Features')
    ax.set_ylabel('R² coefficient')
    ax.set_xticks(ticks)
    ax.set_xticklabels(col_name[:N_features+N],rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    fig.set_figwidth(15)

    plt.savefig(f'plots/non_diag_r2_coefficents_{info}.png')
    plt.savefig(f'plots/non_diag_r2_coefficents__{info}.svg')

    return

def plot_diag_importances(regr_diag,col_name,info):
    importances = regr_diag.feature_importances_

    np.savetxt(f'plots/importances_diag_{info}',importances)

    ticks = np.array(range(len(importances)))

    fig ,ax = plt.subplots()

    ax.bar(ticks ,importances,alpha=1.0, width=0.15, color='orange')
    ax.set_xlabel('Features')
    ax.set_ylabel('R² coefficient')
    ax.set_xticks(ticks)

    ax.set_xticklabels(['x','y','z']+col_name,rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    fig.set_figwidth(15)

    plt.savefig(f'plots/diag_r2_coefficents_{info}.png')
    plt.savefig(f'plots/diag_r2_coefficents_{info}.svg')

    return

def plot_diag_perm_importances(model,X,y,col_name,info):

    importances = permutation_importance(model,X,y,n_repeats=15).importances_mean

    np.savetxt(f'plots/importances_diag_{info}',importances)

    ticks = np.array(range(len(importances)))

    fig ,ax = plt.subplots()

    ax.bar(ticks ,importances,alpha=1.0, width=0.15, color='orange')
    ax.set_xlabel('Features')
    ax.set_ylabel('R² coefficient')
    ax.set_xticks(ticks)

    ax.set_xticklabels(col_name,rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    fig.set_figwidth(15)

    plt.savefig(f'plots/diag_r2_coefficents_{info}.png')
    plt.savefig(f'plots/diag_r2_coefficents_{info}.svg')

    return