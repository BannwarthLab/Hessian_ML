from __future__ import annotations

import gc
import os
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from memory_profiler import profile
from sklearn.model_selection import train_test_split

from hess_ml.src.template import TrainMLHessianGFN2xTB, TrainMLHessianORCA

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
                    TrainMLHessianORCA()
                elif program.lower()  == "xtb":
                    TrainMLHessianGFN2xTB()
            else:
                print("More than one program not implemented yet!")

        self.n_data =  0
        self.splitted = False
        n_split = 0

        max_n_data = 5e5

        for geo in idx:

            self.GenerateData(dir=self.folders[geo])

            print(f"Number of DataPoints {self.n_data}")

            if self.n_data > max_n_data and self.config.general.split_feature:

                self.Targets = np.array(self.Targets)
                self.Features = np.array(self.Features).astype(np.float32)

                print(f"Length of features:{self.Features.shape}")

                with open(f"Features{n_split}.npy","wb") as f:
                    np.save(f,self.Features)
                    f.close()

                with open(f"Targets{n_split}.npy","wb") as f:
                    np.save(f,self.Targets)
                    f.close()

                self.Features = []
                self.Targets = []

                self.n_data =  0
                n_split += 1
                self.splitted = True

        if self.splitted:

            self.Targets = np.array(self.Targets)
            self.Features = np.array(self.Features).astype(np.float32)

            with open(f"Features{n_split}.npy","wb") as f:
                np.save(f,self.Features)
                f.close()

            with open(f"Targets{n_split}.npy","wb") as f:
                np.save(f,self.Targets)
                f.close()

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

        mol = TrainMLHessianGFN2xTB()
        mol.setConfiguration(dir,self.config.general, self.config.molecule)
        mol.ProcessData()

        self.n_data += mol.N_atoms*(mol.N_atoms-1)/2

        print(len(self.Features))

        if mol.do_calc:
            self.Features.extend(mol.get_feature())
            self.Targets.extend(mol.get_target())

        else:
            self.not_considered.append(
                os.path.join(
                    self.config.molecule.folder,
                    self.config.molecule.xyz_file,
                ),
            )

    @profile
    def GenerateData2(self:Environment, dir):
        mol = TrainMLHessianGFN2xTB()
        mol.setConfiguration(dir,self.config.general, self.config.molecule)
        mol.ProcessData()

        self.n_data += mol.N_atoms*(mol.N_atoms-1)/2

        print(len(self.Features))

        return mol.get_feature(),mol.get_target()

    def truncate_file(self, file):
        if os.path.isfile(file):
            with open(file, "wb") as f1:
                f1.truncate(0)

            f1.close()
