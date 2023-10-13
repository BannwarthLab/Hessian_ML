import os

import numpy as np

import time as time

from sklearn.model_selection import train_test_split

from hess_ml.src.Template import TrainMLHessianGFN2xTB
import pickle as pickle
import glob as glob
import copy
class DataGeneration:

    def __init__(self) -> None:
        return

    def generate_data(self, idx=None):
        self.wall_time0 = time.time()

        print("Starting Data Generation for Features...")

        # ________Parallelized Feature Generation___________

        # if idx == None:
        #    idx = np.arange(0,len(self.geo_dir))
        self.not_considered = []
        self.molgen = TrainMLHessianGFN2xTB()
        for geo in idx:
            self.GenerateData(dir=self.folders[geo])

        print("")
        print(
            f"Features and Targets of {len(idx)} structures were generated in {round(time.time() - self.wall_time0)} s\n"
        )

        with open("not_considered", "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close

        return
    
    def GenerateData(self,dir):

        mol = copy.deepcopy(self.molgen)
        mol.setConfiguration(dir, self.config["molecule"])
        mol.ProcessData()

        print(np.array(mol.Feature_AB).shape)
        
        np.savetxt(
            fname=os.path.join(mol.folder, "features"), X=mol.Feature_AB
        )
        np.savetxt(
            fname=os.path.join(mol.folder, "targets"), X=mol.Target_AB
        )

        if mol.do_calc:

            self.Features.extend(mol.Feature_AB)
            self.Targets.extend(mol.Target_AB)
            del mol
        else:
            self.not_considered.append(
                os.path.join(
                    self.config["molecule"]["folder"],
                    self.config["molecule"].get("xyz_file"),
                )
            )
            del mol 


        return 

    def truncate_file(self, file):
        if os.path.isfile(file):
            with open(file, "wb") as f1:
                f1.truncate(0)

            f1.close()

        return

    def do_preparation_split(
        self, folders, total_structures, train_size, test_size, rnd_seed
    ):
        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """
        self.folders = folders

        geo_idx = np.arange(0, total_structures - 1)

        if type(train_size) == list:
            train_size_temp = max(train_size)
        else:
            train_size_temp = train_size

        if train_size_temp == 1.0:
            self.train_idx = geo_idx
            self.test_idx = []

        else:
            self.train_idx, self.test_idx = train_test_split(
                geo_idx,
                test_size=test_size,
                train_size=train_size_temp,
                random_state=rnd_seed,
            )

            self.comp_idx = np.concatenate((self.train_idx, self.test_idx), axis=None)

            geo_idx = geo_idx[self.comp_idx]

        self.test_geo = []

        for i in self.test_idx:
            self.test_geo.append(self.folders[i])

        self.train_geo = []

        for i in self.train_idx:
            self.train_geo.append(self.folders[i])

        self.data_to_txt(self.test_geo, os.path.join("", "test_files.txt"))

        self.data_to_txt(self.train_geo, os.path.join("", "train_files.txt"))

        return
