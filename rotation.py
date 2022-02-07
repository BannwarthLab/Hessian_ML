from cmath import pi
from code import interact
from configparser import InterpolationSyntaxError
from distutils.errors import LinkError
from doctest import DocFileCase
from email import header
from email.errors import HeaderMissingRequiredValue
from operator import matmul
from posixpath import split
from xml.dom import INDEX_SIZE_ERR
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from math import log10 , floor

bohr2angs = 0.52917721067
speed_of_light = 2.9979e10   # in cm/s
mass_unit_in_au = 1.66054e-27 / 9.1094e-31
atomic_time_unit = 2.4189e-17   # E_h / hbar

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
     
def check_eig_vec(eig_vec,coord_check):
     if linalg.det(eig_vec) < 0.:
          for i in range(3):
               eig_vec[i,2] = -eig_vec[i,2]
     return eig_vec

def coord_rot(coord_var,rotM):
     for i in range(len(coord_var.iloc[:,1])):
          coord_var.iloc[i,1:] = matmul(rotM,coord_var.iloc[i,1:])
     return coord_var

def euler_rotation_matrix(alpha,beta,gamma):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    q0 = np.cos(1/2*beta)*np.cos(1/2*(gamma+alpha))
    q1 = np.sin(1/2*beta)*np.sin(1/2*(gamma-alpha))
    q2 = np.sin(1/2*beta)*np.cos(1/2*(gamma-alpha))
    q3 = np.cos(1/2*beta)*np.sin(1/2*(gamma+alpha))
    # Extract the values from Q
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
#R = euler_rotation_matrix(alpha,beta,gamma)

def import_coord(file):
     with open(file) as myfile:
          head = [next(myfile) for x in range(2)]

     coord_var = pd.read_csv(file,sep = '\s+',skiprows = 2,header = None)
     coord_var.columns= ['atoms','x','y','z']
     return coord_var,head

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
          coord_var.iloc[i,1:] = coord_var.iloc[i,1:] - trans
     return coord_var


########## Import
# 
file_path = 'tests/benzol/'
input_path_coord = f'{file_path}'+'benzol.xyz'
output_path_coord = f'{file_path}'+'delete.xyz'

input_path_hess = f'{file_path}'+'hessian_benzol'
output_path = f'{file_path}'+'delete'

coord,head = import_coord(input_path_coord)

hessian = import_hess(input_path_hess,coord)
#
###########

########### Rotation of coordinates and hessian
# Calculating center of mass 
s = center_mass(coord) 

# Translation of coordinate system
coord = vec_trans(coord,s)

# Calculating moment of inertia
I = inert_tensor(coord)

# Calculating eigenvalues and eigenvectors 
eig_val,eig_vec = linalg.eigh(I)

# Check if the coordinate system is right-handed --> important for chirality
eig_vec = check_eig_vec(eig_vec,coord.copy())

# Rotation of the coordinates
coord = coord_rot(coord,np.transpose(eig_vec))

# Construction of the rotation matrix of the hessian and the rotation
P = rotM_hess(R,coord)
rot_hess = matmul(matmul(P,(hessian)),np.transpose(P))

# Calculating the mass weighted hessian and frequencies
hessian_mass = mass_weighted_hessian(hessian,coord['atoms'])

lamb, Q = linalg.eigh(hessian_mass)

freq = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
############

############ Output
# Save the Frequencies
df_out = pd.DataFrame({'Eigenvalues_hessian_self_h2o' : freq})
df_out.to_csv(output_path,sep = '\t')


# 
f = open(output_path_coord,"w")

f.write(head[0])
f.write(head[1])
f.close()

coord.to_csv(output_path_coord, mode ='a',sep = '\t',header = None , index = False)
############


############ For tests
#

R = np.array([[0.869654,0.493585,0.008666],
               [0.493638,-0.869650,-0.005503],
               [0.004820,0.009063,-0.999947]])


trafo_coord = import_coord(f'{file_path}'+'benzol_trafo.xyz')[0]
trafo_hess =  import_hess(f'{file_path}'+'hessian_trafo',trafo_coord)

hes1 = rot_hess
hes2 = trafo_hess

count = 0
for i in range(len(hes1[:,1])):
     for j in range(len(hes1[1,:])):
          if round(hes1[i,j],4) != round(hes2[i,j],4):
               count += 1
               print(i,j,round_it(hes1[i,j],3), round_it(hes2[i,j],3))

print(count)



hessian_mass_trafo = mass_weighted_hessian(trafo_hess,trafo_coord['atoms'])

hes1 = hessian_mass_trafo

hes2 = hessian_mass

print(np.max(hes1))

lamb, Q = linalg.eigh(hessian_mass_trafo)

freq1 = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))



'''
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

empt = np.array([[-1,0.,0.],
               [0.,-0.0,1.0],
               [0.0,1.0,0.0]])

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