import numpy as np
import pandas as pd
from hess_ml.src.Rotation_func import Rotation_Functions
import numpy as np
import pickle as pickle
import os
import json as json
import faulthandler
from ase.io import read as ase_read


class Input:
    def __init__(self):
        pass

    def import_coord(self, file):
        with open(file) as myfile:
            head = [next(myfile) for _ in range(2)]

        coord_var = pd.read_csv(
            file,
            sep="\s+",
            skiprows=2,
            header=None,
            keep_default_na=False,
            na_values=["_"],
        )

        coord_var.columns = ["atoms", "x", "y", "z"]

        return coord_var, head

    def import_dipm(self, file):
        coord_var = pd.read_csv(file, sep=",")

        return coord_var

    def import_hessian_dftd4(self, file, coord):
        nat3 = len(coord["atoms"]) * 3

        file_path = os.path.join(self.geo_working_dir, "dftd4.json")

        with open(file_path) as f:
            egh = json.load(f)

        hess_dftd4 = np.array(egh.get("hessian")).reshape(nat3, nat3)

        return hess_dftd4

    def import_gradient(self, file):
        with open(file, "rb") as f:
            f.close()

        self.gradient = np.genfromtxt(
            file, skip_header=2 + self.N_atoms, skip_footer=1, loose=True
        )
        # gradient = gradient.flatten()

        return

    def import_wbo(self, file):
        wbo = pd.read_csv(file, names=["at1", "at2", "wbo"], sep="\s+")
        return wbo

    def import_pickle_FT_old(self, file):
        feature = []
        target = []

        i = 0

        with open(f"{file}", "rb") as f:
            while True:
                try:
                    i += 1
                    temp_obj = pickle.load(f)
                    feature.extend(temp_obj["Feature"])
                    target.extend(temp_obj["Target_AB"])
                except EOFError:
                    # print(f'Features and Targets of a total of {i-1} structures are used.\n')
                    break

        return feature, target

    def rd_txt_file(self, file):
        with open(f"{file}", "rb") as f:
            filenames = f.read().splitlines()
        f.close()

        for i in range(len(filenames)):
            filenames[i] = filenames[i].decode("ascii")

        return filenames

    def truncate_file(self, file):
        if os.path.isfile(file):
            with open(file, "wb+") as f:
                f.truncate(0)
            f.close()
        return


class Output:
    def __init__(self):
        pass

    def hessian_to_xtb(self, file, hessian):
        """
        Writes a hessian from a numpy array to a xtb format hessian
        Requires file name and the array
        """

        Nat3 = len(hessian)

        with open(file, "w+") as myfile:
            myfile.write(f"$hessian\n")

            for i in range(Nat3):
                str_list = [f"{x: 10.10f}" for x in hessian[i]]

                for k in range(0, Nat3, 5):
                    sep = "\t"

                    sep = sep.join(str_list[k : k + 5])
                    myfile.write("\t")
                    myfile.write(sep)
                    myfile.write("\n")
        myfile.close()

        return

    def data_to_txt(self, data, file):
        with open(file, "w+") as f:
            for d in data:
                f.write(d)
                f.write("\n")
        f.close()

        return
