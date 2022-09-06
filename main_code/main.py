from locale import normalize

from sklearn.preprocessing import Normalizer
from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *

from joblib import parallel_backend
from joblib import dump, load
import pickle
import os.path
#Current working directory
cwd = os.getcwd()
#cwd = 'tests/main_test/'
#cwd[:-9]+'tests/main_test/'
wall_time_0 = time.time()
print(f'Start Importing Files from {cwd}')

train_rot = False
train_set = False

MSE_list = [[],[]]

mol_sys_idx = []
mol_sys_dirs = sorted(os.listdir(cwd))

'''for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'

    struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])

    if os.path.exists("Systems1"):
        with open('Systems1','r+') as f:
            f.truncate(0)
    #For every structure of every molecular system
    count = 0
    for sys in range(len(struc_sys_dirs[:])):#:#
        #print(f'System {struc_sys_dirs[sys]}')

        system = sys_info(folder=f'{mol_dir}{struc_sys_dirs[sys]}/',molecule=mol,variation=sys)

        system.rot_init_inert()

        system.rot_inert_apf()

        mol_sys_idx.append([mol,sys])

        system.project_hessian(label='xTB')

        with open('Systems1','ab') as f:
            pickle.dump(system,f)

        count +=1 

        if count%50 == 0:
            print(f'{count} Structures are imported. \n Wall time: {round(time.time() - wall_time_0)} s')'''

#rnd_state =100#46#,0,56,29,100,208,42,30,39,51]:
for rnd_state in [46,100,0,56,29]:#
#Gathering all directories of all molecular systems 
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
    
    full_y_pred= []
    full_y_test = []

    with open('Systems','rb') as f:
        for _ in range(7164):
            Systems.append(pickle.load(f))

    mol_sys_idx = []
    for mol in range(1):
        for sys in range(7164):
            mol_sys_idx.append([mol,sys])

    '''
    for sys in range(len(Systems)):
        mol_sys_idx.append([0,sys])
    for i in range(len(mol_sys_idx)):
        if mol_sys_idx[i][0] == test_mol:
            test_idx.append(i)
        else:
            train_idx.append(i)
    '''
    wall_time_1 = time.time()

    print(f'All structures are imported. \nWall time: {round(wall_time_1 - wall_time_0)} s')

    #rnd_state =100#46#,0,56,29,100,208,42,30,39,51]:
    print(f'Generating the Feature vectors.')
    wall_time_2 = time.time()

    test_train_split_idx = np.arange(0,len(mol_sys_idx),1)
    train_idx, test_idx = train_test_split(test_train_split_idx,test_size=0.25,train_size=0.15,random_state=rnd_state)

    print(len(train_idx))

    for file in ['Feature_Vector_Hetero','Feature_Vector_Homo','Target_Vector_Hetero','Target_Vector_Homo']:
        if os.path.exists(file):

            with open(file, 'r+') as f:
                f.truncate(0)

    for i in train_idx:

        X_homo_temp,X_hetero_temp = Systems[i].gen_Feature(label = 'indexed',train_rot=train_rot,train_set =train_set)
        Systems[i].gen_Hessian_vector(train_rot=False,train_set =train_set)

        with open('Feature_Vector_Hetero','ab') as f:
            pickle.dump(X_hetero_temp,f)

        with open('Feature_Vector_Homo','ab') as f:
            pickle.dump(X_homo_temp,f)

        with open('Target_Vector_Hetero','ab') as f:
            pickle.dump(Systems[i].H_AB_vec,f)

        with open('Target_Vector_Homo','ab') as f:
            pickle.dump(Systems[i].H_AA_vec,f)


    X_hetero = []
    y_hetero = []

    X_homo = []
    y_homo = []

    with open('Feature_Vector_Hetero','rb') as f:
        for _ in range(len(train_idx)):
            X_hetero.extend(pickle.load(f))

    with open('Target_Vector_Hetero','rb') as f:
        for _ in range(len(train_idx)):
            y_hetero.extend(pickle.load(f))

    with open('Feature_Vector_Homo','rb') as f:
        for _ in range(len(train_idx)):
            X_homo.extend(pickle.load(f))

    with open('Target_Vector_Homo','rb') as f:
        for _ in range(len(train_idx)):
            y_homo.extend(pickle.load(f))
    
    print(np.shape(X_homo))
    print(np.shape(X_hetero))


    #print(f'Generated the Feature vectors. \nWall time: {round(wall_time_3 -wall_time_2)} s')
    wall_time_3 =  time.time()

    print(f'Fitting the Model.')
    n_estimators_homo = 100
    n_estimators_het = 200
    max_depth_homo = 30
    max_depth_het = 30 

    print(f'Homo. Model uses: {n_estimators_homo} Estimators and \na maximal depth of {max_depth_homo}')

    print(f'Het. Model uses: {n_estimators_het} Estimators and \na maximal depth of {max_depth_het}')

    regr_homonuclear=  ExtraTreesRegressor(n_estimators = n_estimators_homo,random_state=rnd_state,bootstrap=True,max_depth=max_depth_homo)
    

    regr_heteronuclear = ExtraTreesRegressor(n_estimators = n_estimators_het,random_state=rnd_state,bootstrap=True,max_depth=max_depth_het)

    #regr_homonuclear = KNeighborsRegressor(n_neighbors=10, weights='distance' )
    #regr_heteronuclear = KNeighborsRegressor(n_neighbors=10, weights='distance' )
    with parallel_backend('threading', n_jobs=14):
        regr_homonuclear.fit(X_homo,y_homo)

    dump(regr_homonuclear,f'ETR_HOMO{rnd_state}.joblib')

    print(f'Homonuclear model fitted.')
    print(f'Number of features seen:{regr_homonuclear.n_features_in_}')

    regr_homonuclear = []

    with parallel_backend('threading', n_jobs=14):
        regr_heteronuclear.fit(X_hetero,y_hetero)
    

    print(f'Heteronuclear model fitted.')
    print(f'Number of features seen:{regr_heteronuclear.n_features_in_}')

    dump(regr_heteronuclear,f'ETR_HETERO{rnd_state}.joblib')
    #####Test part

    regr_heteronuclear = []
    regr_homonuclear = []

    X_hetero = []
    y_hetero = []

    X_homo = []
    y_homo = []

    #Systems = []   
    if True:
        break
    #with open('Systems','rb') as f:
    #    for _ in range(7164):
    #        Systems.append(pickle.load(f))

    regr_homonuclear = load(f'ETR_HOMO{rnd_state}.joblib')
    regr_heteronuclear = load(f'ETR_HETERO{rnd_state}.joblib')
    '''
    with open(f'ETR_HETERO{rnd_state}.json','rb') as f:
        print(f)
        regr_heteronuclear= pickle.loads(f)

    with open(f'ETR_HOMO{rnd_state}.json','rb') as f:
        regr_homonuclear = pickle.loads(f)
    '''

    wall_time_4 = time.time()

    print('Testing the Model.')

    test_sys = None

    for i in test_idx:
        X_homo_temp,X_hetero_temp = Systems[i].gen_Feature(label = 'indexed')
        Systems[i].gen_Hessian_vector()

        full_y_test.extend((Systems[i].hessian).flatten())

        y_homo_pred = regr_homonuclear.predict(X_homo_temp)

        y_hetero_pred = regr_heteronuclear.predict(X_hetero_temp)

        Systems[i].get_pred_hessian(hessian_homo=y_homo_pred,hessian_hetero=y_hetero_pred,check =False)

        full_y_pred.extend((Systems[i].H_pred).flatten())

        Systems[i].weight_hessian(label='xTB')
        Systems[i].weight_hessian(label='pred')

        Systems[i].gen_eigenvalues()
        
        lambd[0].extend(sorted(Systems[i].hessian_lambd))
        freq = frequency(Systems[i].hessian_lambd)
        ZPVE[0].append(np.sum(freq)/2)

        freq_list[0].extend(sorted(freq.copy()))

        lambd[1].extend(sorted(Systems[i].H_pred_lambd))
        freq = frequency(Systems[i].H_pred_lambd)
        ZPVE[1].append(np.sum(freq)/2)

        freq_list[1].extend(sorted(freq.copy()))
        Systems[i].clear_all()

    wall_time_5 = time.time()
    print(f'Testing is done.\n Wall time: {round(wall_time_5 - wall_time_0)} s')

    MSE  = mean_squared_error(full_y_test,full_y_pred)
    MSE_ZPVE = mean_squared_error(np.array(ZPVE[1])*627.5,np.array(ZPVE[0])*627.5)
    MSE_list[0].append(MSE)

    print(f'RMSE :{round(np.sqrt(MSE),5)}')
    print(f'RMSE of ZPVE :{round(np.sqrt(MSE_ZPVE),5)}')

    plt.loglog( [0,max([max(ZPVE[0]),max(ZPVE[1])])*1.1],[0,max([max(ZPVE[0]),max(ZPVE[1])])*1.1],'k-')
    plt.loglog(np.array(ZPVE[0]) ,np.array(ZPVE[1]),'x')

    np.savetxt(f'lambda_ML_Delta{rnd_state}.txt',lambd[1])
    np.savetxt(f'lambda_xTB_Delta{rnd_state}.txt',lambd[0])

    np.savetxt(f'lambda_ML_Delta{rnd_state}.txt',lambd[1])
    np.savetxt(f'lambda_xTB_Delta{rnd_state}.txt',lambd[0])

    np.savetxt(f'ZPVE_ML_Delta{rnd_state}.txt',ZPVE[1])
    np.savetxt(f'ZPVE_xTB_Delta{rnd_state}.txt',ZPVE[0])

    np.savetxt(f'Freq_ML_Delta{rnd_state}.txt',freq_list[1])
    np.savetxt(f'Freq_xTB_Delta{rnd_state}.txt',freq_list[0])

np.savetxt(f'MSE_Hessian_Elements.txt',MSE_list[0])

#np.savetxt(f'MSE_Hetero.txt',MSE_list[1])