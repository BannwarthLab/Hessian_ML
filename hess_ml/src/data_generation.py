from __future__ import annotations

import copy
import os
import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split

from hess_ml.src.template import TrainMLHessianGFN2xTB, TrainMLHessianORCA
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hess_ml.src.environment import Environment

class DataGeneration:
    def __init__(self) -> None:
        return

    def generate_data(self:Environment, idx=None):
        self.wall_time0 = time.time()

        print("Starting Data Generation for Features...")

        # ________Parallelized Feature Generation___________

        # if idx == None:
        #    idx = np.arange(0,len(self.geo_dir))
        self.not_considered = []
        for program in self.config.molecule.program:
            if len(self.config.molecule.program) == 1:
                if program.lower()  == "orca":
                    self.molgen = TrainMLHessianORCA()
                elif program.lower()  == "xtb":
                    self.molgen = TrainMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        

        self.n_data =  0
        self.splitted = False 
        self.n_split = 0

        for geo in idx:
            self.GenerateData(dir=self.folders[geo])

            if self.n_data > 2.8e6:

                self.Targets = np.array(self.Targets)
                self.Features = np.array(self.Features).astype(np.float32)

                np.savetxt(f"Features{self.n_split}.txt", self.Features)
                np.savetxt(f"Targets{self.n_split}.txt", self.Targets)

                self.Features = []
                self.Targets = []
                self.n_data =  0
                self.n_split += 1 
                self.splitted = True 

        if self.splitted:
            print("""Due to the large size of the Features and Targets, the data was split. 
                  No training and predicting is performed as the data is only partially stored in the RAM.""")
            self.config.general.train = False 
            self.config.general.predict = False
            sys.exit()


        print("")
        print(
            f"Features and Targets of {len(idx)} structures "
            f"were generated in {round(time.time() - self.wall_time0)} s\n",
        )

        outputfile_name = "not_considered"
        with open(outputfile_name, "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close()

    def GenerateData(self:Environment, dir):
        mol = copy.deepcopy(self.molgen)
        mol.setConfiguration(dir,self.config.general, self.config.molecule)
        mol.ProcessData()

        self.n_data = mol.N_atoms*(mol.N_atoms-1)/2

        print(np.array(mol.Feature_AB).shape)

        if mol.do_calc:
            self.Features.extend(mol.Feature_AB)
            self.Targets.extend(mol.Target_AB)
            del mol
        else:
            self.not_considered.append(
                os.path.join(
                    self.config.molecule.folder,
                    self.config.molecule.xyz_file,
                ),
            )
            del mol


    def truncate_file(self, file):
        if os.path.isfile(file):
            with open(file, "wb") as f1:
                f1.truncate(0)

            f1.close()

    def do_preparation_split(
        self:Environment,
        folders,
        total_structures,
        train_size,
        test_size,
        rnd_seed,
    ):
        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """

        max_train_size = 1.0
        self.folders:list = folders

        geo_idx = np.arange(0, total_structures)

        train_size_temp = max(train_size) if isinstance(train_size,list) else train_size

        if train_size_temp == max_train_size:
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
