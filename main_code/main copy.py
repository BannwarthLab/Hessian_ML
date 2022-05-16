from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *

#Current working directory
cwd = os.getcwd()

#Gathering all directories of all molecular systems 
mol_sys_dirs = os.listdir(cwd)

#For every molecular system
R_MI_APF_l = []
H_APF_l = []

for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    struc_sys_dirs = glob.glob(f'{cwd}/{mol_sys_dirs[mol]}/*/')

    #For every structure of every molecular system
    for sys in range(len(struc_sys_dirs)):

        coord_init,head   = import_coord(f'{struc_sys_dirs[sys]}init_coord/coord.xyz')
        dipm_init    = import_dipm(f'{struc_sys_dirs[sys]}init_coord/xyz_dipm.csv')
        hessian_init = import_hessian(f'{struc_sys_dirs[sys]}init_coord/hessian',coord_init)


        dipm_init = dipm_init.iloc[:,:-3]


        ############
        ########### Rotation of coordinates and hessian into intermediate position
        # Calculating center of mass 
        s = center_mass(coord_init) 

        # Translation of coordinate system int center of mass
        coord_mc = vec_trans(coord_init,s)

        #vec_trans(dipm,s)
        # Calculating moment of inertia
        I = inert_tensor(coord_mc)

        # Calculating eigenvalues and eigenvectors 
        eig_val,eig_vec = linalg.eigh(I)

        # Check if the coordinate system is right-handed --> important for chirality

        eig_vec = check_eig_vec(eig_vec)

        # Rotating eigenvectors, so that highest values are positive

        eig_vec = eig_vec_rot(eig_vec)

        # Rotation of the coordinates and atomic dipole moments into main inertia system
        coord_MI = coord_rot(coord_mc,eig_vec.copy())

        dipm_MI = coord_rot(dipm_init,eig_vec.copy())

        # Construction of the rotation matrix of the hessian and the rotation
        P_init_mi = rotM_hess(eig_vec.copy(),coord_MI)

        hess_MI = matmul(matmul(P_init_mi,hessian_init.copy()),np.transpose(P_init_mi))
        print(f'Molecule No. {mol} \nSystem No.{sys}')

        

        #Matrix filled with 3x3 rotation matrices for MI to APF
        R_MI_APF_mat = np.zeros([3*len(coord_MI['atoms']),3*len(coord_MI['atoms'])])
        #Matrix filled with hessian for each APF
        H_APF_mat = np.zeros([3*len(coord_MI['atoms']),3*len(coord_MI['atoms'])])

        for atom_A in range(len(coord_MI['atoms'])):
            for atom_B in range(atom_A,len(coord_MI['atoms'])):
                print(f'Atoms: {coord_MI.iloc[atom_A,0]} {atom_A} and {coord_MI.iloc[atom_B,0]} {atom_B}')

                R_MI_APF = get_R_euler(coord_MI,dipm_MI,atom_A,atom_B)

                H_APF = np.zeros([3,3])
                
                #Generate the final hessian
                i0 = 3*atom_A
                i3 = 3*atom_A + 3
                j0 = 3*atom_B 
                j3 = 3*atom_B + 3

                H_APF = matmul(matmul(R_MI_APF,hess_MI[i0:i3,j0:j3].copy()),(np.transpose(R_MI_APF)))

                H_APF_mat[i0:i3,j0:j3] = H_APF
                R_MI_APF_mat[i0:i3,j0:j3] = R_MI_APF

        H_APF_l.append(H_APF_mat)
        R_MI_APF_l.append(R_MI_APF_mat)