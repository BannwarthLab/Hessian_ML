from functions import *
from operator import matmul
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from scipy.spatial.transform import Rotation as rot_trafo
from math import log10 , floor
#############################
#File Path
file_path = 'tests/benzol2/'

init_path_coord = f'{file_path}'+'init_coord/'+'coord.xyz'


#############################
#Import Files
coord,head = import_coord(init_path_coord)
ml_feature = pd.read_csv('ml_feature.csv')

##############################
#Extract features which need to be transformed

    #dipm_atom = ml_feature.loc[:,('dipm_atom_x','dipm_atom_y','dipm_atom_z')]
    #delta_dipm = ml_feature.loc[:,('dipm_delta_x','dipm_delta_y','dipm_delta_z')]
qm_atom  = ml_feature.loc[:,('qm_atom_xx','qm_atom_yy','qm_atom_zz','qm_atom_xy','qm_atom_zx','qm_atom_yz')]
qm_delta = ml_feature.loc[:,('qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_zx','qm_delta_yz')]
    #delta_dipm_mu = ml_feature.loc[:,('delta dipm only mull x','delta dipm only mull y','delta dipm only mull z')]
    #delta_qm_mu = ml_feature.loc[:,('delta qm only mull xx',' delta qm only mull yy',' delta qm only mull zz')]
    #delta_qm_Z = ml_feature.loc[:,('delta qm only Z xx','delta qm only Z yy',' delta qm only Z zz')]


#############################
#Transform to vector by QM x I

qm_atom_mat = np.zeros([len(coord),3])
qm_delta_mat = np.zeros([len(coord),3])

I = np.array([1,1,1])
for i in range(len(qm_atom.iloc[:,0])):
    qm_atom_mat[i,:] = matmul(qm_matrix(qm_atom.iloc[i,:]),I)
    qm_delta_mat[i,:] = matmul(qm_matrix(qm_delta.iloc[i,:]),I)

#############################
#
