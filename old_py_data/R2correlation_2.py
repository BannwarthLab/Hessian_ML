from functions import *
from operator import matmul
from xml.etree import ElementInclude
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


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def box_plot(X,y,thresh,file):
    def comp_var_r2(X,y):
        variances = np.var(X, axis=0)

        from scipy.stats import linregress 

        lin_regs = [linregress(xs, ys) for xs, ys in zip(X.T, [y for i in range(X.shape[1])])]
        rvalues = np.array([ccc.rvalue for ccc in lin_regs])
        return variances,rvalues

    variances, rvalues = comp_var_r2(X,y)

    #ax.plot(variances, label='Variance', c='black')
    plt.bar(np.array(range(len(X[0]))),rvalues ** 2, color='orange')

    #ax.legend(ncol=1, fontsize=10)
    #ax.set_ylim(-0.02, 1.0)
    plt.xlabel('Features')
    plt.ylabel('R² coefficient')
    #ax.set_xticks( np.array(range(len(X[0]))))
    #ax.set_xticklabels(list,rotation=90)
    #print(np.where(rvalues**2 > thresh))
    plt.savefig(file)
    plt.show()
    return

#############################
#Init File Import for Information
file_path = glob.glob('h2/H2_*/')
init_path_coord = f'{file_path[0]}'+'coord.xyz'
coord,head = import_coord(init_path_coord)
Nat = len(coord.iloc[:,0])

#############################
#Import Files
X_all = np.zeros([4,41])
for file in range(len(file_path)):
    init_path_coord = f'{file_path[file]}'+'coord.xyz'
    print(f'Features of {file_path[file]}')
    #############################
    #Import Files
    coord,head = import_coord(init_path_coord)
    Nat = len(coord.iloc[:,0])
    ml_feature = pd.read_csv(f'{file_path[file]}' +'ml_feature.csv')

    ##############################
    #Extract features

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

    #############################
    #Transform to vector by QM x I
    qm_atom_mat = np.zeros([len(coord),3])
    qm_delta_mat = np.zeros([len(coord),3])
    
    I = np.array([1,1,1])

    
    for k in range(len(qm_atom.iloc[:,0])):
        qm_atom_mat[k,:] = matmul(qm_matrix(qm_atom.iloc[k,:]),I)
        qm_delta_mat[k,:] = matmul(qm_matrix(qm_delta.iloc[k,:]),I)


    qm_atom_mat_df = pd.DataFrame(qm_atom_mat,columns = ['qm_x','qm_y','qm_z'])
    qm_delta_mat_df = pd.DataFrame(qm_delta_mat,columns = ['qm_delta_x','qm_delta_y','qm_delta_z'])

    col_name = (CN.columns.values.tolist() + dipm.columns.values.tolist() + qm_atom_mat_df.columns.values.tolist()+ qm_delta_mat_df.columns.values.tolist() + qm.columns.values.tolist() + energy_based.columns.values.tolist())
    for i in range(1):       
        for j in range(1):
            if i == j:                                                    
                    #############################
                    #
                    R_ij = (coord.iloc[i,3] - coord.iloc[j,3])
                    X_i = CN.iloc[i,:]
                    X_i = np.append(X_i,dipm.iloc[i,:])
                    X_i = np.append(X_i,qm_atom_mat_df.iloc[i,:])
                    X_i = np.append(X_i,qm_delta_mat_df.iloc[i,:])
                    X_i = np.append(X_i,qm.iloc[i,:])
                    X_i = np.append(X_i,energy_based.iloc[i,:])
                    """
                    X_j = CN.iloc[j,:]
                    X_j = np.append(X_j,dipm.iloc[j,:])
                    X_j = np.append(X_j,qm_atom_mat_df.iloc[j,:])
                    X_j = np.append(X_j,qm_delta_mat_df.iloc[j,:])
                    X_j = np.append(X_j,qm.iloc[j,:])
                    X_j = np.append(X_j,energy_based.iloc[j,:])

                    X_arith_mean = (X_i + X_j)/2
                    X_prod = X_i * X_j
                    X_abs_diff = np.abs(X_i - X_j)
                    """

                    X = R_ij
                    X = np.append(X,X_i)
                    """
                    X = np.append(X,X_j)
                    X = np.append(X,X_arith_mean)
                    X = np.append(X,X_prod)
                    X = np.append(X,X_abs_diff)
                    """
        X_all[file,:] = X

hess_ij = np.zeros([3])

y_all = np.zeros([9,len(file_path)])
n = 0
for i in range(3):
    for j in range(3):
        if i <= j:
            hess_ij = np.zeros([3])
            hess_ij[i] += 1
            hess_ij[j] += 1
        print(hess_ij)

            for file in range(len(file_path)):
                hessian = import_hess(f'{file_path[file]}' + 'hessian',coord)
                y_all[n,file] = hessian[i,j]
            n += 1

rvals = np.zeros([9,41])

for num in range(9):
    for i in range(len(X_all.T)):
        rvals[num,i] = (linregress(X_all.T[i,:],y_all[num,:]).rvalue)**2


list = ['R_ij'] + col_name

A = 15
B = 41
fig, ax = plt.subplots()
#ax.plot(variances, label='Variance', c='black')
color_list = ['orange','limegreen','firebrick','limegreen','navy','steelblue','firebrick','steelblue','olivedrab']
label_list = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']

ticks = np.array(range(len(X_all[0,A:B])))*3


for i in range(9):
    ax.bar(ticks +i*0.20,rvals[i,A:B],  alpha=1.0, width=0.15, label = label_list[i], color=color_list[i])

#ax.bar(np.array(range(len(X_all[0,:])))+0*0.20,rvals[0,:],  alpha=1.0, width=0.40, color=color_list[0])
#ax.bar(np.array(range(len(X_all[1,:]))),rvals[1,:],  alpha=1.0, width=0.40, color=color_list[1])



ax.legend(ncol=1, fontsize=10)
#ax.set_ylim(-0.02, 1.0)
ax.set_xlabel('Features')
ax.set_ylabel('R² coefficient')
ax.set_xticks(ticks+ 1/2* 6*0.2 )
ax.set_xticklabels(list[A:B],rotation=90)

#print(np.where(rvalues**2 > thresh))

plt.gcf().subplots_adjust(bottom=0.45)
fig.set_figwidth(15)
#plt.savefig('Last.svg')
#plt.show()
                
