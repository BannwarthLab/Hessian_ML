from cmath import pi
from distutils.errors import LinkError
from doctest import DocFileCase
from email import header
from email.errors import HeaderMissingRequiredValue
from operator import matmul
from posixpath import split
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict

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

def center_charge(coord):
     d = np.zeros(3)
     charge_sum = 0
     for i in range(len(coord['atoms'])):
          charge = ELEMENTS2Z[coord.loc[i,'atoms']]
          d += charge*coord.iloc[i,1:]
          charge_sum += charge
     C = d/charge_sum
     return C

def import_hess(file,coord):
     LineList = []
     with open (file,'r') as fd:
          Lines = [line.rstrip('\n') for line in fd]
          for line in Lines[1:]:
               LineList += line.split()

     hess = np.zeros([len(coord['atoms'])*3,len(coord['atoms'])*3])
     i = 0
     for k in range(len(hess[1,:])):
          for l in range(len(hess[:,1])):
               hess[k,l] = float(LineList[i])
               i+=1
     return hess

def import_coord(file):
     with open(file) as myfile:
          head = [next(myfile) for x in range(2)]

     coord = pd.read_csv(file,sep = '\s+',skiprows = 2,header = None)
     coord.columns= ['atoms','x','y','z']
     return coord,head


def vec_trans(vec,trans):
     return vec - trans

def coord_rot(coord,rotM):
     for i in range(len(coord.iloc[:,1])):
          coord.iloc[i,1:] = matmul(matmul(rotM,coord.iloc[i,1:]),rotM)
     return coord

def rotM_hess(R):
     P = np.zeros([3*len(coord['atoms']),3*len(coord['atoms'])])
     for i in range(len(coord['atoms'])):
          P[3*i:3*(i+1),3*i:3*(i+1)] = R
     return P

input_path_coord = 'tests/coord.xyz'

output_path_coord = 'tests/coord_rot.xyz'

input_path_hess = 'tests/hessian_h2o'
output_path = 'tests/out_h2o_rot.txt'


coord,head = import_coord(input_path_coord)

hessian = import_hess(input_path_hess,coord)

shift = [14,2,19]
print(coord)
coord.iloc[:,1:] = coord.iloc[:,1:].add(shift)
print(coord)

s = center_charge(coord)

for i in range(len(coord.iloc[:,1])):
     coord.iloc[i,1:] = vec_trans(coord.iloc[i,1:],s)

R = np.array([ [-1.0,0.0,0.0],
               [0.0,-0.0,1.0],
               [0.0,1.0,0.0]])

coord = coord_rot(coord,R)

print(coord)

P = rotM_hess(R)

rot_hess = matmul(matmul(P,hessian),P)

trafo_coord = import_coord('tests/new.xyz')[0]
trafo_hess =  import_hess('tests/hessian_h2o_rot',trafo_coord)

hes1 = rot_hess
hes2 = trafo_hess
for i in range(len(hes1[:,1])):
     for j in range(len(hes1[1,:])):
          if round(hes1[i,j],4) != round(hes2[i,j],4):
               print(i,j,round(hes1[i,j],4), round(hes2[i,j],4))


f = open(output_path_coord,"w")

f.write(head[0])
f.write(head[1])
f.close()

coord.to_csv(output_path_coord, mode ='a',sep = '\t',header = None , index = False)
