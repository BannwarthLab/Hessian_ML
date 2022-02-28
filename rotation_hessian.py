from operator import matmul
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

def angle_two_vec(a,b):
     cosangle = matmul(a,b)/linalg.norm(a)/linalg.norm(b)
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

def vector_rot(coord_var,rotM):
     for i in range(len(coord_var.iloc[:,1])):
          coord_var.iloc[i,:] = matmul(rotM,coord_var.iloc[i,:])
     return coord_var

#############################
#File Path
file_path = ''

init_path_coord = f'{file_path}'+'init_coord/'+'coord.xyz'

#
#############################
#Import Files
coord,head = import_coord(init_path_coord)
P = np.genfromtxt(file_path+'P_init_inert')


apf = file_path + 'apf_coord/'
H_euler = np.zeros([len(coord.iloc[:,1])*3,len(coord.iloc[:,1])*3])

for i in range(len(coord.iloc[:,1])):
     for j in range(len(coord.iloc[:,1])):
          if i <= j :
               directory = f'atoms_{i}_{j}{coord.iloc[i,0]}{coord.iloc[j,0]}/'

               R_euler = np.genfromtxt(apf+directory+'R_inert_apf.txt')

               i0 = 3*i
               i3 = 3*i + 3
               j0 = 3*j 
               j3 = 3*j + 3
               #############################
               #Actual rotation of the matrix
               H_euler[i0:i3,j0:j3] =matmul(matmul(np.transpose(R_euler),np.genfromtxt(apf+directory+'hessian.txt')[i0:i3,j0:j3]),R_euler)#import_hess(apf+directory+'hessian').iloc[i0:i3,j0:j3]
               if i != j:
                    H_euler[j0:j3,i0:i3] = np.transpose(H_euler[i0:i3,j0:j3])

H_euler = matmul(matmul(np.transpose(P),H_euler),P)
np.savetxt(file_path+'hessian_ML',H_euler)
#
#############################
#
#H = import_hess(file_path +'init_coord/'+'hessian',coord)
#count = 0
#for i in range(len(H[0,:])):
#     for j in range(len(H[0,:])):
#          if round(H[i,j],10)!= round(H_euler[i,j],10):
#               count += 1 
#print(count)



