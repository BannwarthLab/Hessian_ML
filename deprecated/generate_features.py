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

with open('Systems.json','rb') as f:
    Systems = pickle.load(f)

for sys in range(len(Systems)):
    mol_sys_idx.append([0,sys])
'''
for i in range(len(mol_sys_idx)):
    if mol_sys_idx[i][0] == test_mol:
        test_idx.append(i)
    else:
        train_idx.append(i)'''

        #test_size = 1 - train
        #train_size = train
        #print(f'Train size:{train_size}')
        #train_idx, temp = train_test_split(train_idx,test_size=test_size,train_size=train_size,random_state=rnd_state)

wall_time_1 = time.time()
print(f'All structures are imported. \nWall time: {round(wall_time_1 - wall_time_0)} s')

rnd_state = 46#,0,56,29,100,208,42,30,39,51]:
print(f'Generating the Feature vectors.')
wall_time_2 = time.time()

test_train_split_idx = np.arange(0,len(mol_sys_idx),1)
train_idx, test_idx = train_test_split(test_train_split_idx,test_size=0.25,train_size=0.75,random_state=rnd_state)

X_homo = []
y_homo = []

X_hetero = []
y_hetero = []

for i in train_idx:

    Systems[i].gen_Hessian_vector(train_rot=False,train_set =train_set)
    X_homo_temp,X_hetero_temp = Systems[i].gen_Feature(label = 'indexed',train_rot=train_rot,train_set =train_set)

    X_homo.extend(X_homo_temp)
    y_homo.extend(Systems[i].H_AA_vec)

    X_hetero.extend(X_hetero_temp)
    y_hetero.extend(Systems[i].H_AB_vec)

    Systems[i].clear_all()

wall_time_3 =  time.time()

print(len(X_homo),len(X_homo[0]))

np.savetxt('Feature_Vector_Homo',X_homo)
np.savetxt('Target_Vecotor_Homo',y_homo)
np.savetxt('Feature_Vector_Hetero',X_hetero)
np.savetxt('Target_Vector_Hetero',y_hetero)

np.savetxt('train_idx',train_idx)
np.savetxt('test_idx',test_idx)

print(f'Generated the Feature vectors. \nWall time: {round(wall_time_3 -wall_time_2)} s')