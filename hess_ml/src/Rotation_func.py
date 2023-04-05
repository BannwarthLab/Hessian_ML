from scipy import linalg
from operator import matmul
from math import log10 , floor

import numpy as np
from src.constants import Const

class Rotation_Functions:
     def __init__(self):
          pass

     def angle_two_vec(self,a,b):

          if linalg.norm(a) == 0. or linalg.norm(b) == 0.:
               cosangle = 0
          else:
               cosangle = matmul(a,b)/linalg.norm(a)/linalg.norm(b)

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

     def vec_trans(self,coord_var,trans):
          coord_var_new = coord_var.copy()
          for i in range(len(coord_var.iloc[:,1])): 
               coord_var_new.iloc[i,1:] = np.array(coord_var.iloc[i,1:]) - np.array(trans)
          return coord_var_new
     
     def get_artificial_coord(self,coord, aktiv):
     
          r0 = {'H': 0.32, 'D': 0.32, 'He': 0.46, 'Li': 1.20, 'Be': 0.94, 'B': 0.77, 'C': 0.75, 'N': 0.71, 'O': 0.63,
                    'F': 0.64,
                    'Ne': 0.67, 'Na': 1.40, 'Mg': 1.25, 'Al': 1.13, 'Si': 1.04, 'P': 1.10, 'S': 1.02, 'CL': 0.99, 'Ar': 0.96,
                    'K': 1.76, 'Ca': 1.54, 'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22, 'Mn': 1.19, 'Fe': 1.16, 'Co': 1.11,
                    'Ni': 1.10, 'Cu': 1.12, 'Zn': 1.18, 'Ga': 1.24, 'Ge': 1.21, 'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
                    'Rb': 2.10, 'Sr': 1.85, 'Y': 1.63, 'Zr': 1.54, 'Nb': 1.47, 'Mo': 1.38, 'Tc': 1.28, 'Ru': 1.25, 'Rh': 1.25,
                    'Pd': 1.20, 'Ag': 1.28, 'Cd': 1.36, 'In': 1.42, 'Sn': 1.40, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
                    'Cs': 2.32, 'Ba': 1.96, 'La': 1.80, 'Ce': 1.63, 'Pr': 1.76, 'Nd': 1.74, 'Pm': 1.73, 'Sm': 1.72, 'Eu': 1.68,
                    'Gd': 1.69, 'Tb': 1.68, 'Dy': 1.67, 'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64, 'Yb': 1.70, 'Lu': 1.62, 'Hf': 1.52,
                    'Ta': 1.46, 'W': 0.95 * 1.37, 'Re': 1.31, 'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23, 'Au': 1.24, 'Hg': 1.33,
                    'Tl': 1.44,
                    'Pb': 1.44, 'Bi': 1.51, 'Po': 1.45, 'At': 1.47, 'Rn': 1.42, 'Fr': 2.23, 'Ra': 2.01, 'Ac': 1.86, 'Th': 1.75,
                    'Pa': 1.69, 'U': 1.70, 'Np': 1.71, 'Pu': 1.72, 'Am': 1.66, 'Cm': 1.66, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65,
                    'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61, 'Rf': 1.57, 'Db': 1.49, 'Sg': 1.43, 'Bh': 1.41, 'Hs': 1.34,
                    'Mt': 1.29, 'Ds': 1.21, 'Rg': 1.21, 'Cn': 1.22, 'Nh': 1.36, 'Fl': 1.43, 'Mc': 1.62, 'Lv': 1.75, 'Ts': 1.65,
                    'Og': 1.57, 'Cl': 0.99}

          coord_new = np.zeros(3)

          cn = 0
          cn_sum =  0

          for i in range(coord.shape[0]):
               if i != aktiv:

                    x = coord.iloc[i, 1] - coord.iloc[aktiv, 1]
                    y = coord.iloc[i, 2] - coord.iloc[aktiv, 2]
                    z = coord.iloc[i, 3] - coord.iloc[aktiv, 3]

                    r = np.linalg.norm(np.array([x, y, z]))
                    rcov_c = float(r0[f'{coord.iloc[aktiv,0]}'])
                    rcov_i = float(r0[f'{coord.iloc[aktiv,0]}'])
                    rr = (rcov_i + rcov_c) / r

                    cn = 1 / (1 + np.exp( - 16 * (rr - 1)))

                    coord_new += cn*coord.iloc[i, 1:]
                    cn_sum += cn
     

          return coord_new/cn_sum


     #____Uses for i=j the mean of the xyz's atoms as an artifical atom____
     def get_R_euler(self,coord_end,dipm,i,j):
     #Translation
          axis = np.identity(3)

          if i < j:
               T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
               vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]#np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])#

          elif i == j:
               center_CN = self.get_artificial_coord(coord_end,i)#np.sum(coord_end.iloc[:,1:])/len(coord_end.iloc[:,1:])#
               T = 1/2 * coord_end.iloc[i,1:] + center_CN # + 1/2 * coord_end.iloc[j,1:]
               vec_dipm = dipm.iloc[i,1:]#np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])+

          coord_end = self.vec_trans(coord_end,T)

          vec_z = np.zeros(3)
     
          #Rotation for i < j 
          if i < j:
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
          
          # Rotation for i = j
          elif i == j:
               vec_z  = coord_end.iloc[i,1:].astype('float64')
               vec_x = np.cross(vec_z,vec_dipm)

               LL = np.cross(vec_z,axis[2])
               if  np.sum(np.abs(np.array(coord_end.iloc[i,1:3])+center_CN[1:])) < 1e-9:
                    beta = self.angle_two_vec(vec_z,axis[2])
                    alpha = 2*np.pi - self.angle_two_vec(vec_x,axis[0])
                    gamma = 0

                    if linalg.det(np.array([axis[0],axis[2],vec_x])) > 0.:
                         alpha = 2*np.pi - alpha

               else:
                    alpha = self.angle_two_vec(LL,axis[0])
                    beta = self.angle_two_vec(vec_z,axis[2])
                    gamma = self.angle_two_vec(LL,vec_x)

                    #Find right rotation angle
                    if linalg.det(np.array([axis[0],axis[2],LL])) < 0.:
                         alpha = 2*np.pi - alpha
                    
                    if linalg.det(np.array([LL,vec_x,vec_z])) > 0.:
                         gamma = 2*np.pi - gamma 

          R_euler = matmul(matmul(self.rot_Z(gamma),self.rot_X(beta)),self.rot_Z(alpha))

          #Rotation by 180 ° if dipole moment is negative in x

          if matmul(R_euler,vec_x)[0] < 0.:
               vec_z_norm = matmul(R_euler,vec_z)
               vec_z_norm = vec_z_norm/linalg.norm(vec_z_norm)
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
     

     #____Uses for i=j the a damped version of the mean of the xyz's atoms as an artifical atom____

     def get_R_euler_CN(self,coord_end,dipm,i,j):
          #Translation
          axis = np.identity(3)

          if i < j:
               T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
               vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]#np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])#

          elif i == j:


               center_CN = np.sum(coord_end.iloc[:,1:])/len(coord_end.iloc[:,1:])

               
               T = 1/2 * coord_end.iloc[i,1:] + center_CN # + 1/2 * coord_end.iloc[j,1:]
               vec_dipm = dipm.iloc[i,1:]#np.sum(dipm.iloc[:,1:])/len(dipm.iloc[:,1:])+

          coord_end = self.vec_trans(coord_end,T)

          vec_z = np.zeros(3)
     
          #Rotation for i < j 
          if i < j:
               #Atom pair focussed coordinate system
               vec_z  = coord_end.iloc[i,1:].astype('float64')
               vec_x = np.cross(vec_z,vec_dipm)

               LL = np.cross(vec_z,axis[2])

               if  np.sum(np.abs(np.array(coord_end.iloc[[i,j],1:3]))) < 1e-9:
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
          
          # Rotation for i = j
          elif i == j:
               vec_z  = coord_end.iloc[i,1:].astype('float64')
               vec_x = np.cross(vec_z,vec_dipm)

               LL = np.cross(vec_z,axis[2])
               if  np.sum(np.abs(np.array(coord_end.iloc[i,1:3])+center_CN[1:])) < 1e-9:
                    beta = self.angle_two_vec(vec_z,axis[2])
                    alpha = 2*np.pi - self.angle_two_vec(vec_x,axis[0])
                    gamma = 0

                    if linalg.det(np.array([axis[0],axis[2],vec_x])) > 0.:
                         alpha = 2*np.pi - alpha

               else:
                    alpha = self.angle_two_vec(LL,axis[0])
                    beta = self.angle_two_vec(vec_z,axis[2])
                    gamma = self.angle_two_vec(LL,vec_x)

                    #Find right rotation angle
                    if linalg.det(np.array([axis[0],axis[2],LL])) < 0.:
                         alpha = 2*np.pi - alpha
                    
                    if linalg.det(np.array([LL,vec_x,vec_z])) > 0.:
                         gamma = 2*np.pi - gamma 

          R_euler = matmul(matmul(self.rot_Z(gamma),self.rot_X(beta)),self.rot_Z(alpha))

          #Rotation by 180 ° if dipole moment is negative in x

          if matmul(R_euler,vec_x)[0] < 0.:
               vec_z_norm = matmul(R_euler,vec_z)
               vec_z_norm = vec_z_norm/linalg.norm(vec_z_norm)
               R_z = self.rot_Z(np.pi)
               R_euler = matmul(R_z,R_euler)


          #Apply the euler rotation matrix on the new x axis for verification reasons
          vec_x = matmul(R_euler,vec_x)

          coord_end = self.coord_rot(coord_end.copy(),R_euler)

          #Check for Errors in dipole moment or the coordinates
          if vec_x[0] < 0.:
               print(f'Error in vec_x[0] in {i,j}')

          if np.abs(vec_x[1]) > 1e-8 or np.abs(vec_x[2]) > 1e-8:
               print(vec_x)
               print(self.dir)
               print('Error in vec_x')

          if coord_end.iloc[i,1] > 1e-8 or coord_end.iloc[i,2] > 1e-8:
               print('Error in coord i') 

          if coord_end.iloc[j,1] > 1e-8 or coord_end.iloc[j,2] > 1e-8:
               print('Error in coord j')

          return R_euler

     def get_R_euler_old(self,coord_end,dipm,i,j):
          #Translation
          axis = np.identity(3)

          T = 1/2 * coord_end.iloc[i,1:] + 1/2 * coord_end.iloc[j,1:]
          
          coord_end = self.vec_trans(coord_end,T)

          vec_z = np.zeros(3)
          vec_dipm = dipm.iloc[i,1:] + dipm.iloc[j,1:]
          #Rotation for i < j 
          if i < j:
               #Atom pair focussed coordinate system
               vec_z  = coord_end.iloc[i,1:].astype('float64')
               vec_x = np.cross(vec_z,vec_dipm)

               LL = np.cross(vec_z,axis[2])

               if  np.sum(np.abs(np.array(coord_end.iloc[[i,j],1:3]))) < 1e-9:
                    beta = self.angle_two_vec(vec_z,axis[2])
                    alpha = 2*np.pi - self.angle_two_vec(vec_x,axis[0])
                    gamma = 0

                    if linalg.det(np.array([axis[0],axis[2],vec_x])) > 0.:
                         alpha = 2*np.pi - alpha

               else:
                    alpha = self.angle_two_vec(LL,axis[0])
                    beta = self.angle_two_vec(vec_z,axis[2])
                    gamma = self.angle_two_vec(LL,vec_x)

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
               gamma = self.angle_two_vec(LL,vec_x)
               
               #Find right rotation angle
               if vec_x[1]>= 0. : #is this still right??
                    gamma = 2*np.pi - gamma

          #Euler Rotationmatrix

          R_euler = matmul(matmul(self.rot_Z(gamma),self.rot_X(beta)),self.rot_Z(alpha))

          #Rotation by 180 ° if dipole moment is negative in x

          if matmul(R_euler,vec_x)[0] < 0.:
               vec_z_norm = matmul(R_euler,vec_z)
               vec_z_norm = vec_z_norm/linalg.norm(vec_z_norm)
               R_z = self.rot_Z(np.pi)
               R_euler = matmul(R_z,R_euler)


          #Apply the euler rotation matrix on the new x axis for verification reasons
          vec_x = matmul(R_euler,vec_x)

          coord_end = self.coord_rot(coord_end.copy(),R_euler)

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
     


     def rot_Z(self,alpha):
          R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [0,          0,                 1]])
          return R

     def rot_X(self,alpha):
          R = np.array([[1,             0,              0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
          return R 

     def rot_Y(self,alpha):
          R = np.array([ [ np.cos(alpha), 0.0,-np.sin(alpha)],
                         [ 0.0          ,1   , 0.0],
                         [ np.sin(alpha), 0.0, np.cos(alpha)]])
          return R 
     
     def eig_vec_rot(self,eig_vec):
          for i in [0,1]:
               max_abs_val = max(eig_vec[i].min(), eig_vec[i].max(), key=abs)
               if max_abs_val < 0:
                    eig_vec[i] = -eig_vec[i]
                    eig_vec[i+1] = -eig_vec[i+1]
          return eig_vec


     def inert_tensor(self,coord_var):
          inert_t = np.zeros([3,3])
          rot_state = None
          
          '''
          if sum(coord_var['x']) < 1e-9 or sum(coord_var['y']) < 1e-9 or sum(coord_var['z']) < 1e-9:
               rot_state = True
               rotM = matmul(rot_X(np.pi*random.random()),rot_Z(np.pi*random.random()))
               coord_var = coord_rot(coord_var,rotM)
          '''

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

     def H_Approx(grad):
          return np.outer(grad,grad)

     def H_Delta(H_approx,H_exact):
          return H_exact - H_approx 

     def H_Exact(H_approx,H_delta):
          return H_approx + H_delta  


     def mass_weighted_hessian(hessian, atoms):
          for k in range(len(hessian[1,:])//3):
               for l in range(len(hessian[:,1])//3):

                    mass_n = Const.elements_dict[atoms[k]]
                    mass_m = Const.elements_dict[atoms[l]]

                    hessian[3*k:3*k+3,3*l:3*l+3] =  1/np.sqrt(mass_n*mass_m*Const.mass_unit_in_au**2)*hessian[3*k:3*k+3,3*l:3*l+3]

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