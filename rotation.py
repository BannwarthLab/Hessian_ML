from cmath import pi
from distutils.errors import LinkError
from doctest import DocFileCase
from email import header
from email.errors import HeaderMissingRequiredValue
from posixpath import split
import pandas as pd
import numpy as np


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

input_path = 'tests/benzol.xyz'

output_path = 'tests/coord_translation_benzol.xyz'

with open(input_path) as myfile:
    head = [next(myfile) for x in range(2)]


coord = pd.read_csv(input_path,sep = '\s+',skiprows = 2,header = None)
coord.columns= ['atoms','x','y','z']

coord['x'] = coord['x']+3


with open (input_path,'r') as fd:
     Lines = [line.rstrip('\n') for line in fd]

alpha = 0
beta = 0
gamma = 0

R = euler_rotation_matrix(alpha,beta,gamma)

vec = np.array([[1.0,2.0,3.0],
               [1.0,2.0,3.0]])

M = 1/2*(vec[0,:]+ vec[1,:])

vec_new = vec
for i in range(len(vec_new)):
     vec_new[i,:] = vec[i,:] - M

f = open(output_path,"w")

f.write(head[0])
f.write(head[1])
f.close()

coord.to_csv(output_path, mode ='a',sep = '\t',header = None , index = False)
