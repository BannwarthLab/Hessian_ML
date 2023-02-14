import os 
from mpi4py import MPI
import numpy as np

import pickle as pickle
import time as time

from sklearn.model_selection import train_test_split

from Geometry import Geometry
from Geometry import PickleData
class DataGeneration(Geometry):
    
    def __init__(self) -> None:
        Geometry.__init__
        return
                
    def generate_feature_target_sf(self):
        feature_target_file = ['Feature_Vector_Homo','Target_Vector_Homo','Feature_Vector_Hetero','Target_Vector_Hetero']
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank() 

        self.comm.Barrier()
        if self.rank == 0:
            if os.path.isfile(self.output_file):
                with open(self.output_file,'w') as f:
                    f.truncate(0)
            print(f'Generating Features from {self.folder_data}')

           
        if self.train_test == True:
            total_structures = 0
            molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
            for mol in range(len(molecule_dir)):
                data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])
                geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])
                total_structures +=len(sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))]))

            geo_idx = np.arange(total_structures)

            self.test_idx, self.train_idx = train_test_split(geo_idx,test_size=self.test_size,train_size=self.train_size,random_state=self.rnd_seed)
            np.savetxt('test_idx.txt',self.test_idx)
            np.savetxt('train_idx.txt',self.train_idx)

        #_______Reading_the_names_of_all_folders______
        wall_time0 = time.time()

        self.count = 0
        self.idx_list = []

        molecule_dir = sorted([mol for mol in os.listdir(f'{self.cwd}/{self.folder_data}') if os.path.isdir(os.path.join(f'{self.cwd}/{self.folder_data}',mol))])
        for mol in range(len(molecule_dir)):
            #_______Reading_the_names_of_all_subfolders_______
            data_dir = os.path.join(f'{self.cwd}/{self.folder_data}',molecule_dir[mol])

            geo_dir = sorted([geo for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])

            rest_array = np.arange(len(geo_dir),len(geo_dir)+(self.threads-1)-len(geo_dir)%(self.threads-1))

            for geo in range(len(geo_dir)//(self.threads-1)):
                #_____Rotating_Feature_and_Target_for_each_system_____
                #_____MPI_parallelized_with_mpi4py____________________

                if self.rank > 0:
                    sys_idx = geo*(self.size-1) + (self.rank-1)
                    if not(sys_idx in rest_array):
                        #INIT GEOMETRY
                        #
                        #GENERATE TARGET & FEATURE --> picks dependend on env the right features and target

                        self.gen_data(os.path.join(data_dir,geo_dir[geo]),mol,geo)
                        system = PickleData(self)
                        Feature_Target = [self.Feature_AA,self.Target_AA,self.Feature_AB,self.Target_AB]
                        self.clear_quantities()
                        idx = [self.mol,self.geo]
                        
                    else:
                        system = None

                if self.rank > 0:
                    self.comm.send(system,dest=0,tag=0)
                    self.comm.send(idx,dest=0,tag=1)
                    self.comm.send(Feature_Target,dest=0,tag=2)

                else:
                    for proc in range(1,self.size):
                        system_temp = self.comm.recv(source=proc,tag=0)
                        idx_temp = self.comm.recv(source=proc,tag=1)
                        Feature_Target_temp = self.comm.recv(source=proc,tag=2)

                        if system_temp != None:

                            system_temp.add_idx(self.count)

                            with open(self.output_file,'ab') as f:
                                pickle.dump(system_temp,f)

                            if self.count in self.train_idx:
                                for i in range(4):
                                    with open(feature_target_file[i],'ab') as f:
                                        pickle.dump(Feature_Target_temp[i],f)
                                    f.close()

                            self.count +=1
                            self.idx_list.append(idx_temp)

                            if self.count%100 == 0:
                                print(f'{self.count} Structures are imported. Timing: {round(time.time() - wall_time0)} s')
                            
                            del system_temp

        self.comm.Disconnect()
        return
