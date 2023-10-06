import os
from hess_ml.src.IO import Input

from hess_ml.src.Rotation_func import Rotation_Functions
from hess_ml.src.Preparation import Preparation

from hess_ml.src.HessTarget import HessTarget
from hess_ml.src.HessFeature import HessFeature

from hess_ml.src.Observables import Observables
from hess_ml.src.constants import Const

import json as json
import numpy as np
import time as time


class Geometry(
    HessTarget, HessFeature, Input, Preparation, Observables, Rotation_Functions
):
    def __init__(self, folder, config):
        super().__init__

        self.config = config
        self.folder = folder
        self.threads = config.get("threads", 1)
        self.hessian_type = config.get("hessian_type", "vanilla")

        self.xyz_file = config.get("xyz_file", "xtbopt.xyz")
        self.gradient_file = config.get("gradient_file", "gradient")
        self.feature_file = config.get("feature_file", "ml_feature.csv")
        self.target_file = config.get("target_file", "hessian")
        self.hessian_name = config.get("hessian_file", "hessian")
        self.hessian_origin = config.get("hessian_origin", "xtb")
        self.do_calc = True
        
        return

    def gen_data(self, threads):
        if os.path.isfile(os.path.join(self.folder, self.xyz_file)):
            print(self.folder)

            self.xyz_pd, self.header = self.import_coord(
                os.path.join(self.folder, self.xyz_file)
            )

            self.elements = self.xyz_pd["atoms"]

            self.N_atoms = len(self.elements)

            print(f"Number of atoms is {self.N_atoms}")

            if self.N_atoms > 1:
                self.xyz = np.array(self.xyz_pd.iloc[:, 1:])

                self.nuc_charge = np.zeros(self.N_atoms)

                for i in range(self.N_atoms):
                    self.nuc_charge[i] = Const.ELEMENTS2Z[self.elements[i].lower()]

                self.import_ml_features()

                if self.do_calc:
                    cur_time = time.time()

                    self.hessian = self.import_hessian(
                        os.path.join(self.folder, self.target_file), self.N_atoms
                    )

                    self.filter_feature()

                    self.import_gradient(os.path.join(self.folder, self.gradient_file))

                    print(f"Import: {time.time()- cur_time: 4.5f} s")

                    cur_time = time.time()
                    self.rot_inert_apf()
                    print(f"Rotation: {time.time()- cur_time: 0.5f} s")

                    cur_time = time.time()
                    self.get_Feature(threads)
                    print(f"Feature: {time.time()- cur_time: 3.5f} s")

                    cur_time = time.time()

                    self.gen_Hessian_vector(self.transpose_list)
                    print(f"Hessian: {time.time()- cur_time: 3.5f} s")

                    print(
                        np.array(self.Feature_AB).shape, np.array(self.Target_AB).shape
                    )
                    np.savetxt(
                        fname=os.path.join(self.folder, "features"), X=self.Feature_AB
                    )
                    np.savetxt(
                        fname=os.path.join(self.folder, "targets"), X=self.Target_AB
                    )

            else:
                self.do_calc = False

        else:
            print(f"File {os.path.join(self.folder,self.xyz_file)} not found.")
            self.do_calc = False

        return

    def clear_quantities(self):
        del self.cn
        del self.p
        del self.q
        del self.dipm
        del self.qm
        del self.energy_based
        del self.hessian

        return

    def get_feature(self):
        return self.Feature_AB

    def get_target(self):
        return self.Target_AB

    def hessians_difference(self, hess1, hess2):
        self.xyz_pd, self.header = self.import_coord(
            os.path.join(self.folder, self.xyz_file)
        )

        self.elements = self.xyz_pd["atoms"]

        self.N_atoms = len(self.elements)

        self.hessian1 = self.import_hessian(
            os.path.join(self.folder, hess1), self.N_atoms
        )

        self.hessian2 = self.import_hessian(
            os.path.join(self.folder, hess2), self.N_atoms
        )

        self.hess_diff = self.hessian1 - self.hessian2

        return
