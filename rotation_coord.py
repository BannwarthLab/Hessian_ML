from operator import matmul
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from math import log10 , floor
import os
import shutil
from functions import *
bohr2angs = 0.52917721067
speed_of_light = 2.9979e10   # in cm/s
mass_unit_in_au = 1.66054e-27 / 9.1094e-31
atomic_time_unit = 2.4189e-17   # E_h / hbar

if False:
     file_path = 'tests/h2o/'
     input_path_coord = f'{file_path}'+'h2o.xyz'
     output_path_coord = f'{file_path}'

     input_path_hess = f'{file_path}'+'h2o_hess'
     output_path = f'{file_path}'+'rot_h2o_hess'

     input_path_dipm = f'{file_path}'+'xyz_dipm.csv'

     trafo_coord = import_coord(f'{file_path}'+'h2o_oh.xyz')[0]
     trafo_hess =  import_hess(f'{file_path}'+'h2o_oh_hess',trafo_coord)
elif True:
     file_path = ''
     input_path_coord = f'{file_path}'+'init_coord/'+'coord.xyz'

     input_path_hess = f'{file_path}'+'init_coord/'+'hessian'

     input_path_dipm = f'{file_path}'+'init_coord/'+'xyz_dipm.csv'
else:
     file_path = 'tests/benzol/'
     input_path_coord = f'{file_path}'+'benzol.xyz'
     output_path_coord = f'{file_path}'+'benzol_CC.xyz'

     input_path_hess = f'{file_path}'+'benzol_hess'
     output_path = f'{file_path}'+'rot_benzol_hess'

     input_path_dipm = f'{file_path}'+'xyz_dipm.csv'

     trafo_coord = import_coord(f'{file_path}'+'benzol_CC.xyz')[0]
     trafo_hess =  import_hess(f'{file_path}'+'benzol_CC_hess',trafo_coord)

########## Import
# 

coord,head = import_coord(input_path_coord)
hessian = import_hess(input_path_hess,coord)
dipm = import_dipm(input_path_dipm)

dipm = dipm.iloc[:,:-3]
############
########### Rotation of coordinates and hessian into intermediate position
# Calculating center of mass 
s = center_mass(coord) 
# Translation of coordinate system
vec_trans(coord,s)

#vec_trans(dipm,s)
# Calculating moment of inertia
I = inert_tensor(coord)

# Calculating eigenvalues and eigenvectors 
eig_val,eig_vec = linalg.eigh(I)

# Check if the coordinate system is right-handed --> important for chirality

eig_vec = check_eig_vec(eig_vec)

# Rotating eigenvectors, so that highest values are positive

eig_vec = eig_vec_rot(eig_vec)

# Rotation of the coordinates and atomic dipole moments
coord = coord_rot(coord,eig_vec.copy())

dipm = coord_rot(dipm,eig_vec.copy())

# Construction of the rotation matrix of the hessian and the rotation
P = rotM_hess(eig_vec.copy(),coord)

rot_hess = matmul(matmul(P,hessian.copy()),np.transpose(P))

# Calculating the mass weighted hessian and frequencies
hessian_mass = mass_weighted_hessian(rot_hess.copy(),coord['atoms'])

lamb, Q = linalg.eigh(hessian_mass)

freq = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
############


### Saves the coordinates and hessian in the inertia moment axis
#
file_path_inert_CS = f'{file_path}' + 'inert_coord/'
if os.path.exists(file_path_inert_CS):
     shutil.rmtree(file_path_inert_CS)
os.mkdir(file_path_inert_CS)

df_out = pd.DataFrame(rot_hess)
df_out.to_csv(file_path_inert_CS+'hessian',sep = '\t')

f = open(file_path_inert_CS + f'coord.xyz',"w")

f.write(head[0])
f.write(head[1])
f.close()
coord.to_csv(file_path_inert_CS +'coord.xyz', mode ='a',sep = '\t',header = None , index = False)

np.savetxt(f'{file_path}'+'P_init_inert',P)
###

############ Translation and Rotation of the coordinates into end position

# Translation in the center of the bonding



print('Start')

apf = file_path + 'apf_coord/'

if os.path.exists(apf):
     shutil.rmtree(apf)
os.mkdir(apf)

for i in range(len(coord.iloc[:,1])):
     for j in range(len(coord.iloc[:,1])):
          coord_end = coord.copy()
          if i <= j: 
                    print(f'Atoms: {coord_end.iloc[i,0]} {i} and {coord_end.iloc[j,0]} {j}')

                    #Apply the euler rotation matrix on the coordinates
                    # 
                    R_euler = get_R_euler(coord_end,dipm,i,j)       

                    H_euler = np.zeros([len(coord_end.iloc[:,1])*3,len(coord_end.iloc[:,1])*3]) 
                      
                    coord_rot(coord_end,R_euler)

                    #Generate the final hessian

                    i0 = 3*i
                    i3 = 3*i + 3
                    j0 = 3*j 
                    j3 = 3*j + 3
                    
                    rot_hess_ij = rot_hess[i0:i3,j0:j3].copy()

                    H_euler[i0:i3,j0:j3] = matmul(matmul(R_euler,rot_hess_ij),(np.transpose(R_euler)))

                    H_euler[j0:j3,i0:i3] = np.transpose(H_euler[i0:i3,j0:j3])

                    coord_save = coord_end.copy()

                    for k in range(24):
                         rotM_Z = rot_Z(k*15/360*2*np.pi)
                         
                         H_euler[i0:i3,j0:j3] = matmul(matmul(R_euler,H_euler[i0:i3,j0:j3]),(np.transpose(R_euler)))
                         directory = f'atoms_{i}_{j}{coord_end.iloc[i,0]}{coord_end.iloc[j,0]}/'
                         apf_path = os.path.join(apf,directory)

                         if os.path.exists(apf_path):
                              shutil.rmtree(apf_path)

                         os.mkdir(apf_path)

                         np.savetxt(apf_path+'R_inert_apf.txt',R_euler)
                         #np.savetxt(apf_path + 'hessian.txt',H_euler)
                         f = open(apf_path + f'coord.xyz',"w")
                         np.savetxt(apf_path + 'atoms.txt',[i,j])
                         
                         f.write(head[0])
                         f.write(head[1])
                         f.close()

                    coord_save.to_csv(apf_path +'coord.xyz', mode ='a',sep = '\t',header = None , index = False, float_format='{:10.8f}'.format)
