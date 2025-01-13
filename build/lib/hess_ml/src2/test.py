from hess_ml.src2.molecule.molecule import Molecule
from hess_ml.src2.molecule.tblite.prediction_feature import Feature as predFeat
from hess_ml.src2.molecule.tblite.training_feature import Feature as trainFeat
import numpy as np 
path = "/home/guests/gfeldmann/git/hessian_ml/hess_ml/tests/test_files/ethane/"
path = "/home/guests/gfeldmann/projects/hessian/tests/s30l/calc/28/AB"
gname = "xtbopt.xyz"

mol = Molecule(path,gname)
mol.feature = predFeat

mol1 = Molecule(path,gname)
mol1.feature = trainFeat
mol1.read_hessian()

print(np.sum((mol1.feature.processed_features) - mol.feature.processed_features))