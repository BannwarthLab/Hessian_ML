from operator import matmul
from xml.etree import ElementInclude
import pandas as pd
import numpy as np
from mass_charge_dict import ELEMENTS2Z, Z2ELEMENTS,elements_dict
from scipy import linalg
from scipy.spatial.transform import Rotation as rot_trafo
from math import log10 , floor
import os
import shutil

def import_dipm(file):
     coord_var = pd.read_csv(file,sep = ',')
     #coord_var.columns= ['atoms','x','y','z']
     return coord_var

imp = import_dipm('ml_feature_a_low_fhl.csv')


print(np.array(imp['chem.pot (eV)']),np.array(imp['chem_pot_ext']))

print(np.array(imp['HOAO (eV)']),np.array(imp['ehoao_ext']))

print(np.array(imp['LUAO (eV)']),np.array(imp['eluao_ext']))

print(np.array(imp['gap (eV)']),np.array(imp['e_gap_ext']))
#print(imp['e_gap_ext'])