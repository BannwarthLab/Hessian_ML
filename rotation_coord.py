from operator import matmul
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from math import log10 , floor
import os
import shutil
bohr2angs = 0.52917721067
speed_of_light = 2.9979e10   # in cm/s
mass_unit_in_au = 1.66054e-27 / 9.1094e-31
atomic_time_unit = 2.4189e-17   # E_h / hbar

def angle_two_vec(a,b):
     cosangle = np.dot(a,b)
     angle = np.arccos(np.clip(cosangle,-1,1))
     return angle

def center_charge(coord_var):
     d = np.zeros(3)
     charge_sum = 0
     for i in range(len(coord_var['atoms'])):
          charge = ELEMENTS2Z[coord.loc[i,'atoms']]
          d += charge*coord_var.iloc[i,1:]
          charge_sum += charge
     C = d/charge_sum
     return C

def center_mass(coord_var):
     d = np.zeros(3)
     mass_sum = 0
     for i in range(len(coord_var['atoms'])):
          mass = elements_dict[coord_var.loc[i,'atoms']]
          d+= mass*coord_var.iloc[i,1:]
          mass_sum += mass
     M = d/mass_sum
     return M
     
def check_eig_vec(eig_vec):
     if linalg.det(eig_vec) < 0.:
          for i in range(3):
               eig_vec[2,i] = -eig_vec[2,i]
     return eig_vec


def coord_rot(coord_var,rotM):
     for i in range(len(coord_var.iloc[:,1])):
          coord_var.iloc[i,1:] = matmul(rotM,coord_var.iloc[i,1:])
     return coord_var


#R = euler_rotation_matrix(alpha,beta,gamma)


def rot_Z(alpha):
     R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha),  np.cos(alpha), 0],
                   [0,          0,                 1]])
     return R

def rot_X(alpha):
     R = np.array([[1,             0,              0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
     return R 

def eig_vec_rot(eig_vec):
     for i in [0,1]:
          max_abs_val = max(eig_vec[i].min(), eig_vec[i].max(), key=abs)
          if max_abs_val < 0:
               eig_vec[i] = -eig_vec[i]
               eig_vec[i+1] = -eig_vec[i+1]
     return eig_vec

def import_coord(file):
     with open(file) as myfile:
          head = [next(myfile) for x in range(2)]

     coord_var = pd.read_csv(file,sep = '\s+',skiprows = 2,header = None)
     coord_var.columns= ['atoms','x','y','z']
     return coord_var,head

def import_dipm(file):
     coord_var = pd.read_csv(file,sep = ',')
     #coord_var.columns= ['atoms','x','y','z']
     return coord_var

def import_hess(file,coord_var):
     LineList = []
     with open (file,'r') as fd:
          Lines = [line.rstrip('\n') for line in fd]
          for line in Lines[1:]:
               LineList += line.split()

     hess = np.zeros([len(coord_var['atoms'])*3,len(coord_var['atoms'])*3])

     i = 0
     for k in range(len(hess[0,:])):
          for l in range(len(hess[:,0])):
               hess[k,l] = float(LineList[i])
               i+=1
     return hess

def inert_tensor(coord_var):
     inert_t = np.zeros([3,3])
     m = 0
     for i in range(len(coord_var.iloc[:,1])):
          mi = elements_dict[coord_var.iloc[i,0]]
          xi = coord_var.iloc[i,1]
          yi = coord_var.iloc[i,2]
          zi = coord_var.iloc[i,3]
          m += mi

          txx = mi*(yi**2 + zi**2)
          txy = -mi*xi*yi
          txz = -mi*xi*zi
          tyy = mi*(xi**2 + zi**2)
          txz = -mi*xi*zi
          tyz = -mi*yi*zi
          tzz = mi*(yi**2 + xi**2)

          inert_t += np.array([[ txx ,txy  , txz ],
                               [ txy ,tyy , tyz ],
                               [ txz, tyz, tzz ]])

     return inert_t/m/bohr2angs**2

def mass_weighted_hessian(hessian, atoms):
     for k in range(len(hessian[1,:])):
        for l in range(len(hessian[:,1])):
          n = k//3
          m = l//3

          mass_n = elements_dict[atoms[n]]
          mass_m = elements_dict[atoms[m]]

          hessian[k,l] =  1/np.sqrt(mass_n*mass_m*mass_unit_in_au**2)*hessian[k,l]

     return hessian



def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def rotM_hess(R,coord_var):
     P = np.zeros([3*len(coord_var['atoms']),3*len(coord_var['atoms'])])
     for i in range(len(coord_var['atoms'])):
          P[3*i:3*(i+1),3*i:3*(i+1)] = R
     return P

def vec_trans(coord_var,trans):
     for i in range(len(coord_var.iloc[:,1])): 
          coord_var.iloc[i,1:] = np.array(coord_var.iloc[i,1:]) - np.array(trans)
     return coord_var


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
#
###########
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


axis = np.identity(3)

print('Start')

apf = file_path + 'apf_coord/'

if os.path.exists(apf):
     shutil.rmtree(apf)
os.mkdir(apf)

for i in range(len(coord.iloc[:,1])):
     for j in range(len(coord.iloc[:,1])):
          coord_end = coord.copy()
          if i <= j: 
                    H_euler = np.zeros([len(coord.iloc[:,1])*3,len(coord.iloc[:,1])*3]) 

                    #Translation
                    print(f'Atoms: {coord_end.iloc[i,0]} {i} and {coord_end.iloc[j,0]} {j}')

                    T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
                    vec_trans(coord_end,T)

                    vec_z = np.zeros(3)
                    vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]
                    #Rotation for i < j 
                    if i < j:
                         #Atom pair focussed coordinate system
                         vec_z  = (coord_end.iloc[i,1:]).astype('float64')
                         vec_x = np.cross(vec_z,vec_dipm)
                         vec_y = np.cross(vec_z,vec_x)
                         #Euler angles
                         LL = np.cross(vec_z,axis[2])
                         alpha = angle_two_vec(LL,axis[0])
                         beta = angle_two_vec(vec_z,axis[2])
                         gamma = angle_two_vec(LL,vec_x)

                         #Find right rotation angle
                         if linalg.det(np.array([axis[0],axis[2],LL]))<0.:
                              alpha = 2*np.pi - alpha
                         
                         if linalg.det(np.array([LL,vec_x,vec_z])) > 0.:
                              gamma = 2*np.pi - gamma
                         
                    # Rotation for i = j
                    elif i == j:
                         #Atom pair focussed coordinate system
                         vec_z[2] = 1
                         vec_x = np.cross(vec_z,vec_dipm)
                         vec_y = np.cross(vec_z,vec_x)

                         #Euler angles
                         LL = axis[0]
                         beta = 0
                         alpha = 0
                         gamma = angle_two_vec(LL,vec_x)
                         
                         #Find right rotation angle
                         if vec_x[1]>= 0. :
                              gamma = 2*np.pi - gamma

                    #Euler Rotationmatrix

                    R_euler = matmul(matmul(rot_Z(gamma),rot_X(beta)),rot_Z(alpha))

                    #Rotation by 180 ° if dipole moment is negative in x

                    if matmul(R_euler,vec_x)[0] < 0.:
                         vec_z_norm = matmul(R_euler,vec_z)
                         vec_z_norm = vec_z_norm/linalg.norm(vec_z_norm)
                         R_z = rot_Z(np.pi)
                         R_euler = matmul(R_z,R_euler)

                    #Apply the euler rotation matrix on the coordinates                         
                    coord_rot(coord_end,R_euler)

                    #Apply the euler rotation matrix on the new x axis for verification reasons
                    vec_x = matmul(R_euler,vec_x)

                    #Check for Errors in dipole moment or the coordinates
                    if vec_x[0] < 0.:
                         print(f'Error in vec_x[0] in {i,j}')

                    if np.abs(vec_x[1]) > 1e-8 or np.abs(vec_x[2]) > 1e-8:
                         print(vec_x)
                         print('Error in vec_x')

                    if coord_end.iloc[i,1] > 1e-8 or coord_end.iloc[i,2] > 1e-8:
                         print('Error in coord i') 

                    if coord_end.iloc[j,1] > 1e-8 or coord_end.iloc[j,2] > 1e-8:
                         print('Error in coord j') 


                    #Generate the final hessian

                    i0 = 3*i
                    i3 = 3*i + 3
                    j0 = 3*j 
                    j3 = 3*j + 3

                    rot_hess_ij = rot_hess[i0:i3,j0:j3].copy()

                    H_euler[i0:i3,j0:j3] = matmul(matmul(R_euler,rot_hess_ij),(np.transpose(R_euler)))

                    H_euler[j0:j3,i0:i3] = np.transpose(H_euler[i0:i3,j0:j3])

                    coord_save = coord_end.copy()

                    directory = f'atoms_{i}_{j}{coord_end.iloc[i,0]}{coord_end.iloc[j,0]}/'
                    apf_path = os.path.join(apf,directory)

                    if os.path.exists(apf_path):
                         shutil.rmtree(apf_path)

                    os.mkdir(apf_path)

                    np.savetxt(apf_path+'R_inert_apf.txt',R_euler)
                    #np.savetxt(apf_path + 'hessian.txt',H_euler)
                    f = open(apf_path + f'coord.xyz',"w")

                    print(coord_save)
                    f.write(head[0])
                    f.write(head[1])
                    f.close()
                    coord_save.to_csv(apf_path +'coord.xyz', mode ='a',sep = '\t',header = None , index = False, float_format='{:10.8f}'.format)


# -0.0482709677   0.0000134249  0.0000826599 
# -0.0000146129  -0.0961939722  0.0395004333
#  0.0000811338  -0.0394967983 -0.3326545804 

# -0.0482816600   0.0008710819  -0.0006293365
#  0.0008429957  -0.0961787538   0.0394983713
#  0.0007930118  -0.0394863365  -0.3326545889

#       0.000000    -0.000000     1.000000
#      0.607833     0.794065     0.000000
#     -0.794065     0.607833     0.000000


#np.savetxt('heuler.txt',H_euler)         
#np.savetxt('tests/h2o/P_matrix.txt',Ph)

#np.savetxt('tests/h2o/hess_test.txt',H_euler)
              
############


############ Output
# Save the Frequencies



# 
"""
f = open(output_path_coord,"w")

f.write(head[0])
f.write(head[1])
f.close()

coord_save.to_csv(output_path_coord, mode ='a',sep = '\t',header = None , index = False)
"""
############


############ For tests
#
'''


hes1 = rot_hess
hes2 = trafo_hess

count = 0
for i in range(len(hes1[:,1])):
     for j in range(len(hes1[1,:])):
          if round(hes1[i,j],4) != round(hes2[i,j],4):
               count += 1
               #print(i,j,round_it(hes1[i,j],3), round_it(hes2[i,j],3))




hessian_mass_trafo = mass_weighted_hessian(trafo_hess,trafo_coord['atoms'])

hes1 = hessian_mass_trafo

hes2 = hessian_mass

lamb, Q = linalg.eigh(hessian_mass_trafo)

freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))


shift = [14,2,19]
print(coord)
coord.iloc[:,1:] = coord.iloc[:,1:].add(shift)
print(coord)


     rot = np.identity(3)
     coord_check = coord_rot(coord_check,eig_vec)
     if len(coord_check.iloc[:,1]) > 1:

          temp = coord_check.iloc[0,1:]
          
          if (temp[0]*temp[1]*temp[2]> 0.0):
               for i in range(3):
                    if (temp[i] < 0):
                         rot[i,i] = -1
'''

"""
I = [[0.121048,0.,0.],
     [0., 0.232615 ,0.],
     [0.,0., 0.353663]]
"""

"""
eig_vec = np.array([[-1,0.,0.],
               [0.,-0.0,1.0],
               [0.0,1.0,0.0]])
"""

"""
empt = np.array([[-1,0.,0.],
               [0.,-0.0,1.0],
               [0.0,1.0,0.0]])
"""

"""
Check if first coord is all positive
coord_check = coord.copy()
rot = np.identity(3)
if len(coord_check.iloc[0,:]) > 1:
     temp = coord_check.iloc[0,1:]
     for i in range(3):
          if (temp[i] < 0.00):
               rot[i,i] = -1


empt = np.array([[-1,0.,0.],
               [0.,-0.0,1.0],
               [0.0,1.0,0.0]])

coord_final = coord_rot(coord.copy(),rot)
"""