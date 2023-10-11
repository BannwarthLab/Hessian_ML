import pandas as pd 
import numpy as np 
import os 
import json 


def importXYZ(file):

    with open(file) as myfile:
        head = [next(myfile) for _ in range(2)]

    xyz = pd.read_csv(
        file,
        sep="\s+",
        skiprows=2,
        header=None,
        keep_default_na=False,
        na_values=["_"],
    )

    xyz.columns = ["atoms", "x", "y", "z"]
    
    
    return xyz, head

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
