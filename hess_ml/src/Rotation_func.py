from scipy import linalg
from operator import matmul
from math import log10 , floor

import numpy as np
from hess_ml.src.constants import Const

class Rotation_Functions:
     def __init__(self):
          pass

     def angle_two_vec(self,a,b):

          if linalg.norm(a) == 0. or linalg.norm(b) == 0.:
               cosangle = 0
          else:
               cosangle = np.dot(a,b)/linalg.norm(a)/linalg.norm(b)

          angle = np.arccos(np.clip(cosangle,-1,1))
          
          return angle

     def center_charge(self,coord_var):
          d = np.zeros(3)
          charge_sum = 0
          for i in range(len(coord_var['atoms'])):
               charge = Const.ELEMENTS2Z[coord_var.loc[i,'atoms']]
               d += charge*coord_var.iloc[i,1:]
               charge_sum += charge
          C = d/charge_sum
          return C

     def center_mass(self,coord_var):
          d = np.zeros(3)
          mass_sum = 0
          for i in range(len(coord_var['atoms'])):
               mass = Const.elements_dict[coord_var.loc[i,'atoms']]
               d+= mass*coord_var.iloc[i,1:]
               mass_sum += mass
          M = d/mass_sum
          return M
          
     def check_eig_vec(self,eig_vec):
          if linalg.det(eig_vec) < 0.:
               for i in range(3):
                    eig_vec[2,i] = -eig_vec[2,i]
          return eig_vec


     def coord_rot(self,coord_var,rotM):
          for i in range(len(coord_var.iloc[:,1])):
               coord_var.iloc[i,1:] = matmul(rotM,coord_var.iloc[i,1:])
          return coord_var
     
     def rot_gradient(self,R):
          for i in range(self.N_atoms):
               self.gradient[i] = matmul(R,self.gradient[i])
          return

     def vec_trans(self,coord_var,trans):
          coord_var_new = coord_var.copy()
          for i in range(len(coord_var.iloc[:,1])): 
               coord_var_new.iloc[i,1:] = np.array(coord_var.iloc[i,1:]) - np.array(trans)
          return coord_var_new


     #____Uses for i=j the mean of the xyz's atoms as an artifical atom____
     def get_R_euler(self,coord_end,dipm,i,j):

          if i==j:
               print('error')
               
          axis = np.identity(3)

          
          T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
          
          vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]#np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])#

          coord_end = self.vec_trans(coord_end,T)

          vec_z = np.zeros(3)
     
          #Rotation for i < j 

          #Atom pair focussed coordinate system
          vec_z  = coord_end.iloc[i,1:].astype('float64')
          vec_x = np.cross(vec_z,vec_dipm)

          LL = np.cross(vec_z,axis[2])

          if  np.sum(np.abs(np.array(coord_end.iloc[[i,j],1:3]))) < 1e-12:
               beta = self.angle_two_vec(vec_z,axis[2])
               alpha = 2*np.pi - self.angle_two_vec(vec_x,axis[0])
               gamma = 0

               if linalg.det(np.array([axis[0],axis[2],vec_x])) > 0.:
                    alpha = 2*np.pi - alpha

          else:
               alpha = self.angle_two_vec(LL,axis[0])
               beta = self.angle_two_vec(vec_z,axis[2])
               gamma =self.angle_two_vec(LL,vec_x)

               #Find right rotation angle
               if linalg.det(np.array([axis[0],axis[2],LL])) < 0.:
                    alpha = 2*np.pi - alpha
               
               if linalg.det(np.array([LL,vec_x,vec_z])) > 0.:
                    gamma = 2*np.pi - gamma
          
          R_euler = matmul(matmul(self.rot_Z(gamma),self.rot_X(beta)),self.rot_Z(alpha))

          #Rotation by 180 ° if dipole moment is negative in x

          if matmul(R_euler,vec_x)[0] < 0.:

               R_z = self.rot_Z(np.pi)
               R_euler = matmul(R_z,R_euler)


          #Apply the euler rotation matrix on the new x axis for verification reasons
          vec_x = matmul(R_euler,vec_x)

          self.angle_two_vec(LL,vec_x)
          coord_end = self.coord_rot(coord_end.copy(),R_euler)

          #Check for Errors in dipole moment or the coordinates
          if vec_x[0] < 0.:
               print(f'Error in vec_x[0] in {i,j}')

          if np.abs(vec_x[1]) > 1e-8 or np.abs(vec_x[2]) > 1e-8:
               print(vec_x)
               print(f'Error in vec_x for {i,j}')
               print(coord_end.iloc[[i,j],:])
               print(alpha,beta,gamma)

          if coord_end.iloc[i,1] > 1e-8 or coord_end.iloc[i,2] > 1e-8:
               print(f'Error in coord i:{i,j}')
               print(coord_end.iloc[[i,j],:])
               print(alpha,beta,gamma)

          if coord_end.iloc[j,1] > 1e-8 or coord_end.iloc[j,2] > 1e-8:
               print(f'Error in coord j:{i,j}') 
               print(coord_end.iloc[[i,j],:])
               print(alpha,beta,gamma)

          return R_euler
     

     def rot_Z(self,alpha): #Givens rotation around the z-axis
          R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [0,          0,                 1]])
          return R

     def rot_X(self,alpha): #Givens rotation around the x-axis
          R = np.array([[1,             0,              0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
          return R 

     def rot_Y(self,alpha): #Givens rotation around the y-axis
          R = np.array([ [ np.cos(alpha), 0.0,-np.sin(alpha)],
                         [ 0.0          ,1   , 0.0],
                         [ np.sin(alpha), 0.0, np.cos(alpha)]])
          return R 
     
     def eig_vec_rot(self,eig_vec): #Checks for the highest value of the eigenvector matrix exchanges if the highest is not in first place
          for i in [0,1]:
               max_abs_val = max(eig_vec[i].min(), eig_vec[i].max(), key=abs)
               if max_abs_val < 0:
                    eig_vec[i] = -eig_vec[i]
                    eig_vec[i+1] = -eig_vec[i+1]

          return eig_vec


     def inert_tensor(self,coord_var): #computes the inert tensor 
          inert_t = np.zeros([3,3])

          m = 0
          for i in range(len(coord_var.iloc[:,1])):
               mi = Const.elements_dict[coord_var.iloc[i,0]]
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

               inert_t += np.array([[ txx ,txy , txz ],
                                   [ txy ,tyy , tyz ],
                                   [ txz , tyz , tzz ]])

          return inert_t/m/Const.bohr2angs**2

     def H_Delta(H_approx,H_exact):
          return H_exact - H_approx
          
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

     def rotM_hess(self,R,coord_var):
          P = np.zeros([3*len(coord_var['atoms']),3*len(coord_var['atoms'])])
          for i in range(len(coord_var['atoms'])):
               P[3*i:3*(i+1),3*i:3*(i+1)] = R
          return P

     def rotM_hess2(self,R,n):
          P = np.zeros([3*n,3*n])
          for i in range(n):
               P[3*i:3*(i+1),3*i:3*(i+1)] = R
          return P


     def vector_rot(coord_var,rotM):
          coord_var_new = coord_var.copy()
          for i in range(len(coord_var.iloc[:,1])):
               coord_var_new.iloc[i,:] = matmul(rotM,coord_var.iloc[i,:])
          return coord_var_new


     def calc_R(self,coord):
          ############
          ########### Rotation of coordinates and hessian into intermediate position
          # Calculating center of mass 
          s = self.center_mass(coord) 

          # Translation of coordinate system int center of mass
          coord = self.vec_trans(coord,s)
          #vec_trans(dipm,s)

          # Calculating moment of inertia
          I = self.inert_tensor(coord)
          # Calculating eigenvalues and eigenvectors 
          eig_val,eig_vec = linalg.eigh(I)

          # Check if the coordinate system is right-handed --> important for chirality
          eig_vec = self.check_eig_vec(eig_vec)

          # Rotating eigenvectors, so that highest values are positive

          eig_vec = self.eig_vec_rot(eig_vec)

          return eig_vec,coord
     
     def qm_matrix(qm_atom):
          qm_matrix_list = []
          for i in range(len(qm_atom)):
               xx = qm_atom[i,0]
               yy = qm_atom[i,1]
               zz = qm_atom[i,2]
               xy = qm_atom[i,3]
               xz = qm_atom[i,4]
               yz = qm_atom[i,5]

               qm_matrix = np.array([[xx,xy,xz],
                                   [xy,yy,yz],
                                   [xz,yz,zz]])

               qm_matrix_list.append(qm_matrix)

          return qm_matrix_list
     