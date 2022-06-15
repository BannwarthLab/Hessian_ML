from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *


cwd = os.getcwd()
cwd = 'tests/main_test/'
print(f'Start Importing Files from {cwd}')

mol_sys_dirs = sorted(os.listdir(cwd))

train_rot = True

Systems = []
mol_sys_idx = []
for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
    struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])
    #print(f'Molecule No. {mol}')
    if mol_sys_dirs[mol] == 'H3':
        test_mol = mol
    #For every structure of every molecular system
        for sys in range(len(struc_sys_dirs)):
            #print(f'System {struc_sys_dirs[sys]}')

            system = sys_info(folder=f'{mol_dir}{struc_sys_dirs[sys]}/init_coord/',molecule=mol,variation=sys)

            system.rot_init_inert()

            system.rot_inert_apf()

            mol_sys_idx.append([mol,sys])

            Systems.append(system)

            #print('Fitting the Model.')
            rnd_state = None

test_idx = []
train_idx = []

for i in range(len(mol_sys_idx)):
    if mol_sys_idx[i][0] == test_mol:
        test_idx.append(i)
    else:
        train_idx.append(i)

print(Systems[test_idx[3]].get_coord_state())
print(Systems[test_idx[3]].init_R_MI)
print(Systems[test_idx[3]].xyz)

print(Systems[test_idx[6]].get_coord_state())
print(Systems[test_idx[6]].init_R_MI)
print(Systems[test_idx[6]].xyz)