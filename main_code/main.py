from constants import *
from ml_func import *
from rotation_func import *
import os
import glob as glob

#Current working directory
cwd = os.getcwd()

#Gathering all directories of all molecular systems 
mol_sys_dirs = os.listdir(cwd)

for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    struc_sys_dirs = glob.glob(f'{cwd}/{mol_sys_dirs[mol]}/*/')

    for sys in range(len(mol_sys_dirs)):

        

