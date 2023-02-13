from packages import *
from constants import *
from functions import *


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


def get_feature_col_name(ml_feature_directory):
    feature_path = os.path.join(ml_feature_directory)
    feature_df = pd.read_csv(feature_path)
    col = extract_feature(feature_df.iloc[0],'yx')

    return col


def gen_X_y_list(hessian,ml_features,atom_A,atom_B):
    Nat = pd.DataFrame()

    k = 0 
    temp_all = []
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