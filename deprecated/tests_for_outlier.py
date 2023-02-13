from sklearn import neighbors
from constants import *
from ml_func import *
from functions import *
from rotation_func import *
from packages import *
from class_sys_info import *

from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from itertools import cycle, islice
#Current working directory
cwd = os.getcwd()
cwd = 'tests/main_test/'
#cwd[:-9]+'tests/main_test/'

print(f'Start Importing Files from {cwd}')

train_rot = True
train_set = False

MSE_list = [[],[]]

#Gathering all directories of all molecular systems 
mol_sys_dirs = sorted(os.listdir(cwd))
train_set_fraction = [None]# np.linspace(0.2,0.99,8)
#For every molecular system
lambd = [[],[]]
ZPVE = [[],[]]
X_homo_list =  []
X_hetero_list =  []

for train in train_set_fraction:
    Systems = []
    mol_sys_idx = []
    for mol in range(len(mol_sys_dirs)):
        #Gathering for each molecular systems all directories of diffrent structures
        mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
        struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])
        #print(f'Molecule No. {mol}')
        #For every structure of every molecular system
        for sys in range(len(struc_sys_dirs)):
            #print(f'System {struc_sys_dirs[sys]}')

            system = sys_info(folder=f'{mol_dir}{struc_sys_dirs[sys]}/init_coord/',molecule=mol,variation=sys)

            system.rot_init_inert()

            system.rot_inert_apf()

            mol_sys_idx.append([mol,sys])

            Systems.append(system)
            X_homo_temp,X_hetero_temp = Systems[-1].gen_Feature(label = 'indexed',data_analysis=True)

            X_homo_list.append(X_homo_temp[0])
            X_hetero_list.append(X_hetero_temp[0])

X_homo_arr = np.array(X_homo_list)

X_hetero_arr = np.array(X_hetero_list)

scaler = preprocessing.MinMaxScaler().fit(X_hetero_arr)

X_hetero_arr_scaled = scaler.fit_transform(X_hetero_arr)

#plt.scatter(np.arange(0,len(X_hetero_arr_scaled[:,1])),X_hetero_arr_scaled[:,1])

clustering = DBSCAN(eps=1.5,min_samples=6).fit(np.array(X_hetero_arr_scaled))

y_pred =clustering.labels_.astype(int)

train_idx, test_idx = train_test_split(X_hetero_arr,test_size=0.5,train_size=0.5,random_state=0)

lof = LocalOutlierFactor(n_neighbors=5,novelty=True)

lof.fit(train_idx) 

y_pred = lof.predict(test_idx)

colors = np.array(
    list(
        islice(
            cycle(
                [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                    "#FF8A2BE2",
                    "#FFFF7F50",
                    "#FF00008B"
                ]
            ),
            int(max(y_pred) + 1),
        )
    )
)


colors = np.append(colors, ["#000000"])

plt.scatter(np.arange(0,len(test_idx)),test_idx[:,22],color=colors[y_pred])

plt.show()


'''from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X_hetero_arr_scaled)
distances, indices = nbrs.kneighbors(X_hetero_arr_scaled)

# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]


plt.plot(distances,'k.')
#plt.show()'''