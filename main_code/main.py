from numpy import reshape
from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *

#Current working directory
cwd = os.getcwd()
cwd = 'tests/main_test/'
print(f'Start Importing Files from {cwd}')

#Gathering all directories of all molecular systems 
mol_sys_dirs = sorted(os.listdir(cwd))

#For every molecular system
Systems = []
mol_sys_idx = []

for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
    struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])
    print(f'Molecule No. {mol}')
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

print('Fitting the Model.')
rnd_state = None

test_idx = []
train_idx = []
for i in range(len(mol_sys_idx)):
    if mol_sys_idx[i][0] == test_mol:
        test_idx.append(i)
    else:
        train_idx.append(i)

for rnd_state in [42]:#,0,56,29,100]:
    
    test_train_split_idx = np.arange(0,len(mol_sys_idx),1)
    #train_idx, test_idx = train_test_split(test_train_split_idx,test_size=0.25,train_size=0.75,random_state=rnd_state)

    
    X_homo = []
    y_homo = []

    X_hetero = []
    y_hetero = []

    for i in train_idx:

        Systems[i].gen_Hessian_vector(train_rot=True)
        X_homo_temp,X_hetero_temp = Systems[i].gen_Feature(label = 'indexed',train_rot=True)

        X_homo.extend(X_homo_temp)
        y_homo.extend(Systems[i].H_AA_vec)

        X_hetero.extend(X_hetero_temp)
        y_hetero.extend(Systems[i].H_AB_vec)

    regr_homonuclear=  ExtraTreesRegressor(n_estimators = 300,random_state=rnd_state,bootstrap=False) 
    regr_heteronuclear = ExtraTreesRegressor(n_estimators = 300,random_state=rnd_state,bootstrap=False)

    regr_homonuclear.fit(X_homo,y_homo)
    regr_heteronuclear.fit(X_hetero,y_hetero)

    full_y_pred_homo = []
    full_y_pred_hetero = []

    full_X_homo_test = []
    full_X_hetero_test = []

    full_y_test_homo = []
    full_y_test_hetero = []

    print('Testing the Model.')
    lambd = [[],[]]

    for i in test_idx:
        Systems[i].gen_Hessian_vector()
        X_homo_temp,X_hetero_temp = Systems[i].gen_Feature(label = 'indexed')

        if i in [test_idx[1],test_idx[8]]:
            print(f'{i}')
            print(len(Systems[i].features.Feature_AA))
            print(Systems[i].features.Feature_AA[0][0:9])
            print(Systems[i].features.Feature_AA[9][0:9])
            print(Systems[i].features.Feature_AA[18][0:9])
            #print(Systems[i].features.Feature_AB[0][0:9])

        # if i == test_idx[0]:
        #     print(X_homo_temp[0])
        full_X_homo_test.extend(X_homo_temp)
        full_X_hetero_test.extend(X_hetero_temp)

        full_y_test_homo.extend(Systems[i].H_AA_vec)
        full_y_test_hetero.extend(Systems[i].H_AB_vec)

        y_homo_pred = regr_homonuclear.predict(X_homo_temp)
        y_hetero_pred = regr_heteronuclear.predict(X_hetero_temp)
        #y_hetero_pred

        Systems[i].get_pred_hessian(hessian_homo=y_homo_pred,hessian_hetero=y_hetero_pred)

        full_y_pred_homo.extend(y_homo_pred)
        full_y_pred_hetero.extend(y_hetero_pred)

        Systems[i].project_hessian(label='xTB')
        Systems[i].project_hessian(label='pred')

        Systems[i].weight_hessian(label='xTB')
        Systems[i].weight_hessian(label='pred')
        
        lambd[0].append(Systems[i].hessian_lambd)
        lambd[1].append(Systems[i].H_pred_lambd)

    length = np.linspace(0.6,4,10)
    for i in range(len(length)):
        for j in range(4):
            plt.plot(length[i],lambd[1][i][j],'bx')
            plt.plot(length[i],lambd[0][i][j],'rx')



    #print(Systems[test_idx[1]].features.qm_atom[:])
    #print(Systems[test_idx[8]].features.qm_atom[:])
    #print(Systems[test_idx[2]].R_MI_APF_mat-Systems[test_idx[7]].R_MI_APF_mat)

    r_score_homo = regr_homonuclear.score(full_X_homo_test,full_y_test_homo)
    r_score_hetero = regr_heteronuclear.score(full_X_hetero_test,full_y_test_hetero)

    MSE_homonuclear  = mean_squared_error(full_y_test_homo,full_y_pred_homo)
    MSE_heteronuclear  = mean_squared_error(full_y_test_hetero,full_y_pred_hetero)

    print(f'R2(homo)   :{round(r_score_homo,3)}')
    print(f'R2(hetero) :{round(r_score_hetero,3)}')

    print(f'MSE(homo)  :{round(MSE_homonuclear,3)}')
    print(f'MSE(hetero):{round(MSE_heteronuclear,3)}')

    plt.xlabel(r'$x$(H2) [$\mathrm{\AA}$]',fontsize=24)
    plt.ylabel(r'$k~[\mathrm{N~\mathrm{m}^{-1}}]$',fontsize=24)  
    
    plt.show()

    #plt.savefig(f'plots/lambda_symmetry_check{time_atm}.png')
    #plt.savefig(f'plots/lambda_symmetry_check{time_atm}.svg')

