from calendar import c
from msilib.schema import Feature
from turtle import shape

from sklearn import preprocessing
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern ,RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

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
#
file_path = glob.glob('tests/h2/H2_*/') + glob.glob('tests/N2/N2_*/') + glob.glob('tests/O2/O2_*/') +glob.glob('tests/F2/F2_*/')
init_path_coord = f'{file_path[0]}'+'coord.xyz'
coord,head = import_coord(init_path_coord)
Nat = len(coord.iloc[:,0])

#############################
#Import Files

Feature_Coll = np.empty([len(file_path),Nat,40])
Hessian_Coll = np.empty([len(file_path),3*Nat,3*Nat])

for file in range(len(file_path)):
    init_path_coord = f'{file_path[file]}'+'coord.xyz'
    print(f'Features of {file_path[file]}')
    #############################
    #Import Files
    coord,head = import_coord(init_path_coord)
    Nat = len(coord.iloc[:,0])
    ml_feature = pd.read_csv(f'{file_path[file]}' +'ml_feature.csv')
    hessian = import_hess(f'{file_path[file]}' + 'hessian',coord)

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

    X_df = pd.concat([CN,dipm,qm_atom_mat_df,qm_delta_mat_df,qm,energy_based],axis =1)

    ##############################
    #Stores all Hessians and Feature in a ndarray
    Feature_Coll[file] = np.array(X_df)
    Hessian_Coll[file] = hessian


###############################
#Generating the specific feature for diagonal matrix blocks
X = np.zeros([len(file_path)*Nat*(3)**2,43])
y = np.zeros(len(file_path)*Nat*(3)**2)
k = 0
grps_diag = y.copy()
l = 0
for file in range(len(file_path)):
    for A in range(Nat):
        for i in range(3):
            for j in range(3):
                xyz = np.array([0,0,0])
                grps_diag[k] = int(l)
                xyz[i] += 1
                xyz[j] += 1
                X[k,0:3] = xyz
                X[k,3:]  = Feature_Coll[file,A]
                y[k] = Hessian_Coll[file,i+3*A,j+3*A]
                k+=1
        l+=1

#X = preprocessing.normalize(X,norm ='l2')


gss = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state=42)
idx = list(gss.split(X, y, grps_diag))
train_idx = idx[0][0]
test_idx =  idx[0][1]
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

regr_diag = RandomForestRegressor( n_estimators = 100,random_state=42,bootstrap=False)
#regr_diag = GaussianProcessRegressor(kernel = Matern() ,random_state=42)

regr_diag.fit(X_train, y_train)

print('Score of diag model:',regr_diag.score(X_test,y_test))
"""
importances = regr_diag.feature_importances_

ticks = np.array(range(len(X[0])))

fig ,ax = plt.subplots()

ax.bar(ticks ,importances,alpha=1.0, width=0.15, color='orange')

#ax.bar(np.array(range(len(X_all[0,:])))+0*0.20,rvals[0,:],  alpha=1.0, width=0.40, color=color_list[0])
#ax.bar(np.array(range(len(X_all[1,:]))),rvals[1,:],  alpha=1.0, width=0.40, color=color_list[1])


#ax.set_ylim(-0.02, 1.0)
ax.set_xlabel('Features')
ax.set_ylabel('R² coefficient')
ax.set_xticks(ticks )
ax.set_xticklabels(['x','y','z']+col_name,rotation=90)

#print(np.where(rvalues**2 > thresh))

plt.gcf().subplots_adjust(bottom=0.45)
fig.set_figwidth(15)
plt.show()

"""
non_diag_len = (Nat**2-Nat)*3**2
X_non_diag = np.zeros([len(file_path)*non_diag_len,203])
y_non_diag = np.zeros(len(file_path)*non_diag_len)
grps_non_diag = y_non_diag.copy()
l = 0
k = 0
for file in range(len(file_path)):
    for A in range(Nat):
        for B in range(Nat):
            if A != B:
                for i in range(3):
                    for j in range(3):
                            #if A < B:
                            #print('i,j=',i,j)
                            #print('A,B=',A,B)
                            grps_non_diag[k] = int(l)
                            xyz = np.array([0,0,0])
                            xyz[i] += 1
                            xyz[j] += 1

                            X_non_diag[k,0:3] = xyz
                            X_non_diag[k,3:43] = Feature_Coll[file,A]
                            X_non_diag[k,43:83]= Feature_Coll[file,B]
                            X_non_diag[k,83:123]= (Feature_Coll[file,A] +Feature_Coll[file,B])/2
                            X_non_diag[k,123:163]= (Feature_Coll[file,B]*Feature_Coll[file,A])
                            X_non_diag[k,163:203]= np.abs(Feature_Coll[file,B] -Feature_Coll[file,A])


                            y_non_diag[k] = Hessian_Coll[file,i+3*A,j+3*B]
                            k+=1
                l += 1



'''
for k in range(3*Nat):
    for l in range(3*Nat):
            k = i%Nat
            l = j%Nat
            print('k,l=',k,l)
            C = k%3
            D = l%3
            print('C,D=',C,D)
'''
#X_non_diag_train, X_non_diag_test, y_non_diag_train, y_non_diag_test = train_test_split(X_non_diag, y_non_diag, test_size=0.25 ,random_state=42)

#gss_non_diag = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state=42)
#idx_non_diag = list(gss_non_diag.split(X_non_diag, y_non_diag, grps_non_diag))
#train_idx = idx_non_diag[0][0]
#test_idx =  idx_non_diag[0][1]

X_non_diag_train = X_non_diag[train_idx]
X_non_diag_test = X_non_diag[test_idx]

y_non_diag_train = y_non_diag[train_idx]
y_non_diag_test = y_non_diag[test_idx]

regr_non_diag = RandomForestRegressor(n_estimators = 100,random_state=42, bootstrap=False)
#regr_non_diag = GaussianProcessRegressor(kernel = Matern(),random_state=42)

regr_non_diag.fit(X_non_diag_train, y_non_diag_train)

hess_non_diag_pred = regr_non_diag.predict(X_non_diag_test)
hess_diag_pred = regr_diag.predict(X_test)

print('Score of non diag model:',regr_non_diag.score(X_non_diag_test,y_non_diag_test))

num = 18*8
file_num = test_idx[num]//18

hess_diag = np.zeros([3,3])
hess_non_diag = np.zeros([3,3])

k= int(num//9*9)
for i in range(3):
    for j in range(3):

        hess_diag[i,j] = hess_diag_pred[k]
        hess_non_diag[i,j] = hess_non_diag_pred[k]

        k += 1

hess = np.zeros([6,6])


hess[0:3,0:3] = hess_diag
hess[3:6,3:6] = hess_diag
hess[0:3,3:6] = hess_non_diag
hess[3:6,0:3] = hess_non_diag


#P = np.genfromtxt('h2_2/P_init_inert')

init_path_coord = f'{file_path[file_num//2]}'+'coord.xyz'
coord,head = import_coord(init_path_coord)

#hess_pred = matmul(matmul(np.transpose(P),hess),P)
hessian_mass = mass_weighted_hessian(hess,coord['atoms'])
lamb, Q = linalg.eigh(hessian_mass)
freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))

print(freq1)
#hess_test = matmul(matmul(np.transpose(P),Hessian_Coll[5]),P)

hess_test =Hessian_Coll[file_num]
hessian_mass = mass_weighted_hessian(hess_test,coord['atoms'])
lamb, Q = linalg.eigh(hessian_mass)
freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))

print(freq1)

