from packages import *
from constants import *


def project_hess(hess_v,coord):
    idx = find_trans_rot(hess_v.copy(),coord.copy())
    lamb,Q = linalg.eigh(hess_v)
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

    return idx_list


def wavenumber(lamb):
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi*speed_of_light))
    return freq_val

def frequency(lamb):
    freq_val = (np.sqrt(abs(lamb))/(atomic_time_unit*2*np.pi))
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