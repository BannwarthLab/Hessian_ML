from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from scipy.spatial.transform import Rotation as rot_trafo
from math import log10 , floor
import numpy as np
from operator import matmul
import pandas as pd

bohr2angs = 0.52917721067
speed_of_light = 2.9979e10   # in cm/s
mass_unit_in_au = 1.66054e-27 / 9.1094e-31
atomic_time_unit = 2.4189e-17   # E_h / hbar
hplanck = 6.62607015e-34 # hplank Js
conv_J_to_eV = (1.602176634e-19)**-1 #eV/J

def angle_two_vec(a,b):
     cosangle = matmul(a,b)/linalg.norm(a)/linalg.norm(b)
     angle = np.arccos(np.clip(cosangle,-1,1))
     return angle

def center_charge(coord_var):
     d = np.zeros(3)
     charge_sum = 0
     for i in range(len(coord_var['atoms'])):
          charge = ELEMENTS2Z[coord_var.loc[i,'atoms']]
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

def get_R_euler(coord_end,dipm,i,j):
     #Translation
     axis = np.identity(3)

     T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
     
     vec_trans(coord_end,T)

     vec_z = np.zeros(3)
     vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]
     #Rotation for i < j 
     if i < j:

          #Atom pair focussed coordinate system
          
          vec_z  = (coord_end.iloc[i,1:]).astype('float64')
          vec_x = np.cross(vec_z,vec_dipm)
          #vec_y = np.cross(vec_z,vec_x)

          #Euler angles
          LL = np.cross(vec_z,axis[2])

          if  np.sum(np.abs(np.array(coord_end.iloc[[i,j],1:3]))) < 1e-9:
               beta = angle_two_vec(vec_z,axis[2])
               alpha = 2*np.pi - angle_two_vec(vec_x,axis[0])
               gamma = 0
               
               if linalg.det(np.array([axis[0],axis[2],vec_x])) > 0.:
                    alpha = 2*np.pi - alpha

          else:
               alpha = angle_two_vec(LL,axis[0])
               beta = angle_two_vec(vec_z,axis[2])
               gamma = angle_two_vec(LL,vec_x)

               #Find right rotation angle
               if linalg.det(np.array([axis[0],axis[2],LL])) < 0.:
                    alpha = 2*np.pi - alpha
               
               if linalg.det(np.array([LL,vec_x,vec_z])) > 0.:
                    gamma = 2*np.pi - gamma
          
     # Rotation for i = j
     elif i == j:
          #Atom pair focussed coordinate system
          vec_z[2] = 1
          vec_x = np.cross(vec_z,vec_dipm)
          #vec_y = np.cross(vec_z,vec_x)

          #Euler angles
          LL = axis[0]
          beta = 0
          alpha = 0
          gamma = angle_two_vec(LL,vec_x)
          
          #Find right rotation angle
          if vec_x[1]>= 0. : #is this still right??
               gamma = 2*np.pi - gamma

     #Euler Rotationmatrix

     R_euler = matmul(matmul(rot_Z(gamma),rot_X(beta)),rot_Z(alpha))

     #Rotation by 180 ° if dipole moment is negative in x

     if matmul(R_euler,vec_x)[0] < 0.:
          vec_z_norm = matmul(R_euler,vec_z)
          vec_z_norm = vec_z_norm/linalg.norm(vec_z_norm)
          R_z = rot_Z(np.pi)
          R_euler = matmul(R_z,R_euler)


     #Apply the euler rotation matrix on the new x axis for verification reasons
     vec_x = matmul(R_euler,vec_x)

     coord_end = coord_rot(coord_end.copy(),R_euler)

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

     return R_euler

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

     
def qm_matrix(qm_atom,name):

     xx = qm_atom.loc[f'{name}xx']
     xy = qm_atom.loc[f'{name}xy']
     yy = qm_atom.loc[f'{name}yy']
     xz = qm_atom.loc[f'{name}xz']
     zz = qm_atom.loc[f'{name}zz']
     yz = qm_atom.loc[f'{name}yz']

     qm_matrix = np.array([[xx,xy,xz],
                          [xy,yy,yz],
                          [xz,yz,zz]])

     return qm_matrix

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
