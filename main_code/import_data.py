from locale import normalize

from sklearn.preprocessing import Normalizer
from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
import pickle
#Current working directory
cwd = os.getcwd()
#cwd = 'tests/main_test/'
#cwd[:-9]+'tests/main_test/'
wall_time_0 = time.time()
print(f'Start Importing Files from {cwd}')

train_rot = False
train_set = False

MSE_list = [[],[]]

#Gathering all directories of all molecular systems 
mol_sys_dirs = sorted(os.listdir(cwd))
train_set_fraction = [None]# np.linspace(0.2,0.99,8)
#For every molecular system
lambd = [[],[]]
ZPVE = [[],[]]
freq_list = [[],[]]

y_pred_list_homo = []
y_pred_list_hetero = []

y_pred_list_hetero_freq = []
y_pred_list_homo_freq = []
 
Systems = []
mol_sys_idx = []
for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
    struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])
    #print(f'Molecule No. {mol}')
    #if mol_sys_dirs[mol] == 'H3':
     #   test_mol = mol
    #For every structure of every molecular system
    count = 0
    for sys in range(len(struc_sys_dirs)):
        #print(f'System {struc_sys_dirs[sys]}')

        system = sys_info(folder=f'{mol_dir}{struc_sys_dirs[sys]}/',molecule=mol,variation=sys)

        system.rot_init_inert()

        system.rot_inert_apf()

        mol_sys_idx.append([mol,sys])

        Systems.append(system)

        count +=1 

        if count%50 == 0:
            print(f'{count} Structures are imported. \n Wall time: {round(time.time() - wall_time_0)} s')



with open('Systems.json','wb') as f:
    pickle.dump(Systems,f)
