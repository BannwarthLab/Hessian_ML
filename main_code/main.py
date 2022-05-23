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
X_heteronuclear = []
y_heteronuclear = []

X_homonuclear  = []
y_homonuclear = []

Systems = []
mol_idx_heteronuclear = []
mol_idx_homonuclear = []

Target_length_heteronuclear = []
Traget_lentth_homonuclear = []

for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
    struc_sys_dirs = [ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ]
    print(f'Molecule No. {mol}')
    #For every structure of every molecular system
    for sys in range(len(struc_sys_dirs)):
        print(f'System No. {sys}')

        system = sys_info(folder=f'{mol_dir}{struc_sys_dirs[sys]}/init_coord/',molecule=mol,variation=sys)

        system.rot_init_inert()

        #Matrix filled with 3x3 rotation matrices for MI to APF
        R_MI_APF_mat = np.zeros([3*len(system.xyz['atoms']),3*len(system.xyz['atoms'])])

        #Matrix filled with hessian for each APF
        H_APF_mat = np.zeros([3*len(system.xyz['atoms']),3*len(system.xyz['atoms'])])

        system.rot_inert_apf()

        system.gen_Hessian_vector()

        Feature_heteronuclear, Target_heteronuclear, Feature_homonuclear, Target_homonuclear = system.connect_Feature_Target()

        mol_idx_heteronuclear.append(len(y_heteronuclear))
        mol_idx_homonuclear.append(len(y_homonuclear))

        Target_length_heteronuclear.append(len(Target_heteronuclear))
        Traget_lentth_homonuclear.append(len(Target_homonuclear))

        X_heteronuclear.extend(Feature_heteronuclear)
        y_heteronuclear.extend(Target_heteronuclear)

        X_homonuclear.extend(Feature_homonuclear)
        y_homonuclear.extend(Target_homonuclear)

        Systems.append(system)

X_heteronuclear = np.array(X_heteronuclear)
y_heteronuclear = np.array(y_heteronuclear)

X_homonuclear = np.array(X_homonuclear)
y_homonuclear = np.array(y_homonuclear)


def gen_grp(list):
    full_list = []
    for i in range(len(list)):
        for _ in range(list[i]):
            full_list.append(i)
    return full_list


grp_heteronuclear = gen_grp(Target_length_heteronuclear)
grp_homonuclear = gen_grp(Traget_lentth_homonuclear)

rnd_state = 42

gss = GroupShuffleSplit(n_splits=1,train_size=0.75,random_state = rnd_state)

idx_homonuclear = list(gss.split(X_homonuclear, y_homonuclear, grp_homonuclear))
idx_heteronuclear= list(gss.split(X_heteronuclear,y_heteronuclear,grp_heteronuclear))

X_homonuclear_train= X_homonuclear[idx_homonuclear[0][0]]
y_homonuclear_train= y_homonuclear[idx_homonuclear[0][0]]
 
X_homonuclear_test = X_homonuclear[idx_homonuclear[0][1]]
y_homonuclear_test = y_homonuclear[idx_homonuclear[0][1]]

X_heteronuclear_train = X_heteronuclear[idx_heteronuclear[0][0]]
y_heteronuclear_train = y_heteronuclear[idx_heteronuclear[0][0]]

X_heteronuclear_test = X_heteronuclear[idx_heteronuclear[0][1]]
y_heteronuclear_test = y_heteronuclear[idx_heteronuclear[0][1]]


regr_homonuclear=  ExtraTreesRegressor(n_estimators = 300,random_state=rnd_state,bootstrap=False) 
regr_heteronuclear = ExtraTreesRegressor(n_estimators = 100,random_state=rnd_state,bootstrap=False)

regr_homonuclear.fit(X_homonuclear_train,y_homonuclear_train)
regr_heteronuclear.fit(X_heteronuclear_train,y_heteronuclear_train)


y_homonuclear_pred = regr_homonuclear.predict(X_homonuclear_test)

y_heteronuclear_pred = regr_heteronuclear.predict(X_heteronuclear_test)

MSE_heteronuclear  = mean_squared_error(y_homonuclear_test,y_homonuclear_pred)

print(MSE_heteronuclear)