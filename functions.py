from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from scipy.spatial.transform import Rotation as rot_trafo
from math import log10 , floor
import numpy as np
from operator import matmul
import pandas as pd
import glob as glob
import os


bohr2angs = 0.52917721067
speed_of_light = 2.9979e10   # in cm/s
mass_unit_in_au = 1.66054e-27 / 9.1094e-31
atomic_time_unit = 2.4189e-17   # E_h / hbar
atomic_mass_unit = 1.6605390666e-27 # in kg
hplanck = 6.62607015e-34 # hplank Js
conv_J_to_eV = (1.602176634e-19)**-1 #eV/J

def project_hess(hess_v,coord):
    idx,lamb,Q = find_trans_rot(hess_v.copy(),coord.copy())
    hess_projected_v = hess_v.copy()

    for i in idx:
        i = int(i)
        hess_projected_v -= lamb[i] * np.outer(Q.T[i], Q.T[i].T)

    return hess_projected_v,Q

def find_trans_rot(hess,coord):
    Nat = len(coord)

    overlap_mat = np.zeros([6,3*Nat])
    
    trans_x = np.array([1.,0.,0.])
    trans_y = np.array([0.,1.,0.])
    trans_z = np.array([0.,0.,1.])

    for i in range(Nat):
        overlap_mat[0,3*i:3*i+3] = trans_x
        overlap_mat[1,3*i:3*i+3] = trans_y
        overlap_mat[2,3*i:3*i+3] = trans_z

        overlap_mat[3,3*i:3*i+3] = np.array([0.,coord.loc[i,'z'],-coord.loc[i,'y']])
        overlap_mat[4,3*i:3*i+3] = np.array([-coord.loc[i,'z'],0.,coord.loc[i,'x']])
        overlap_mat[5,3*i:3*i+3] = np.array([coord.loc[i,'y'],-coord.loc[i,'x'],0.])

    overlap_mat = overlap_mat / Nat
    overlap_mat = 1/(linalg.norm(overlap_mat,1)) * overlap_mat


    lamb, Q = linalg.eigh(hess)

    M = matmul(overlap_mat,Q)

    norm_x = np.array(linalg.norm(coord.loc[:,'x']))
    
    norm_y = np.array(linalg.norm(coord.loc[:,'y']))

    norm_z = np.array(linalg.norm(coord.loc[:,'z']))

    idx_len = 6 

    if (norm_x + norm_y) < 1e-6 or (norm_y + norm_z) < 1e-6 or (norm_z + norm_x)< 1e-6:
        idx_len = 5

    M_sum = np.zeros(3*Nat)
    for i in range(len(M_sum)):
        M_sum[i] = np.sum(np.abs(M[:,i]))

    idx_list = np.zeros(idx_len)
    for i in range(idx_len):
        idx = np.where(M_sum == np.amax(M_sum))[0][0]
        idx_list[i] = int(idx)
        M_sum[idx] -= M_sum[idx]

    return idx_list,lamb,Q

def wavenumber(lamb):
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
    return freq_val

def frequency(lamb):
     
     freq_val = np.zeros(len(lamb))

     for i in range(len(lamb)):
          
          if lamb[i] >= 0:
               freq_val[i] = np.sqrt(lamb[i])
          else:
               freq_val[i] = 0

     return freq_val


def force_constant(lamb,atoms):
    m_sum = 0
    for i in range(len(atoms)):
        m_sum += 1/elements_dict[atoms[i]]
    mu = 1/m_sum
    fc = mu*lamb
    return fc

def freq_extract(freq):
    freq = freq.copy()
    list_freq = []
    while np.amax(freq) > 1e-3 :
        idx = np.where(freq == np.amax(freq))[0][0]
        list_freq.append(freq[idx])
        freq[idx] = 0

    return list(list_freq)



def gen_full_hess_mat_from_vector(y_df,hess_diag_pred,hess_non_diag_pred,num_atoms,mol,file_num,rot_arr):

    lenH = 3 * int(num_atoms)
    clean_hess = np.zeros([lenH,lenH])
    idx_mol_var = y_df.loc[(y_df['molecule'] == mol) & (y_df['variation'] == file_num) & (y_df['y_idx'] == 'xx')].index.values.tolist()
    mat_list = np.arange(int(num_atoms+(num_atoms**2 - num_atoms)/2))#gen_sym_mat_list(int(num_atoms))

    o = 0
    for l in range(len(idx_mol_var)):

        i = idx_mol_var[l]

        A = int(y_df.loc[i,'atom1'])
        B = int(y_df.loc[i,'atom2'])

        k = mat_list[l]
        if A == B:
            rot = rot_arr[k]
            hess_vec = hess_diag_pred[9*A:9*A+9]

            hess_mat = hess_vec_to_hess_block(hess_vec)
            clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            
        elif A != B:
            hess_vec = hess_non_diag_pred[o:o+9]
            o += 9
            rot = rot_arr[k]
            hess_mat = hess_vec_to_hess_block(hess_vec)

            clean_hess[3*A:3*A+3,3*B:3*B+3] = matmul(matmul(np.transpose(rot),hess_mat),(rot))

            hess_mat = hess_vec_to_hess_block(hess_vec)
            clean_hess[3*B:3*B+3,3*A:3*A+3] = np.transpose(matmul(matmul(np.transpose(rot),hess_mat),(rot)))


    return clean_hess

def get_grps(X_diag_prep,X_non_diag_prep):

    grps_diag = np.array(X_diag_prep['mol_idx'])
    grps_non_diag = np.array(X_non_diag_prep['mol_idx'])

    return grps_diag,grps_non_diag
    
def gen_rot_arr(path_variation):
    path_apf_list = glob.glob(f'{path_variation}'+'apf_coord/atoms_*')
    rot_arr = []
    for path_apf in path_apf_list:
        rot = np.genfromtxt(os.path.join(path_apf,'R_inert_apf.txt'))
        rot_arr.append(rot)
    return rot_arr

def gen_sym_mat_list(N):
    idx_list = []
    k = N
    l = 0
    m = 1
    while k > 0:
        for _ in range(k):
            idx_list.append(l)
            l+= 1 
        o = N -1

        if k >1:

            for _ in range(N-k+1):
                idx_list.append(m)
                m += o
                o-= 1
        m = N-k+2
        k -= 1
    return idx_list

def hess_block_to_hess_vec(hess_mat,A,B):
    k = 0
    hess_vec = np.zeros(9)
    for i in range(3):
        for j in range(3):
            hess_vec[k] =  hess_mat[3*int(A)+i,3*int(B)+j]
            k+=1
    return hess_vec

def hess_vec_to_hess_block(hess_vec):
    k = 0 
    hess_mat = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            hess_mat[i,j] = hess_vec[k]
            k += 1
    return hess_mat



def angle_two_vec(a,b):

     if linalg.norm(a) == 0. or linalg.norm(b) == 0.:
          cosangle = 0
     else:
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

def import_hessian(file,coord_var):
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

#Das kann evtl. noch verschnellert werden, indem man direkt bspw. H_AB mit den Massen multipliziert.
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
     xz = qm_atom.loc[f'{name}zx']
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
