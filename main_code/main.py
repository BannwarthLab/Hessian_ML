from xml.sax.handler import feature_string_interning
from numpy import var
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
    struc_sys_dirs = glob.glob(f'{cwd}\\{mol_sys_dirs[mol]}/*/')

    #For every structure of every molecular system
    for sys in range(len(struc_sys_dirs)):

        system = sys_info(folder=f'{struc_sys_dirs[sys]}\\init_coord\\',molecule=mol,variation=sys)

        system.rot_init_inert()

        print(f'Molecule No. {mol} \nSystem No.{sys}')
        feature = Feature(folder = f'{struc_sys_dirs[sys]}\\init_coord\\')
        #Matrix filled with 3x3 rotation matrices for MI to APF
        R_MI_APF_mat = np.zeros([3*len(system.xyz['atoms']),3*len(system.xyz['atoms'])])
        #Matrix filled with hessian for each APF
        H_APF_mat = np.zeros([3*len(system.xyz['atoms']),3*len(system.xyz['atoms'])])

        for atom_A in range(len(system.xyz['atoms'])):
            for atom_B in range(atom_A,len(system.xyz['atoms'])):
                print(f'Atoms: {system.xyz.iloc[atom_A,0]} {atom_A} and {system.xyz.iloc[atom_B,0]} {atom_B}')

                R_MI_APF = get_R_euler(system.xyz,system.dipm,atom_A,atom_B)

                H_APF = np.zeros([3,3])
                
                #Generate the final hessian
                i0 = 3*atom_A
                i3 = 3*atom_A + 3
                j0 = 3*atom_B 
                j3 = 3*atom_B + 3

                H_APF = matmul(matmul(R_MI_APF,system.hessian[i0:i3,j0:j3].copy()),(np.transpose(R_MI_APF)))

                
                H_APF_mat[i0:i3,j0:j3] = H_APF
                R_MI_APF_mat[i0:i3,j0:j3] = R_MI_APF

        H_APF_l.append(H_APF_mat)

        R_MI_APF_l.append(R_MI_APF_mat)


        
