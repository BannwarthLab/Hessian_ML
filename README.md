A python package providing a framework for the training and using machine learning (ML) models to predict Hessian matrices and compute thermodynamic corrections. 

# Minimal use example:

```
from mlhess.utils.chemistry.molecule import Molecule
from mlhess.machinelearning.target.hessian import NuclearHessianPM
from mlhess.machinelearning.constructor import load_model
from tcgm_lib.trv.trv_models.mrrho import TruhlarCramerRRHO

model = load_model()
mol = Molecule("/path/to/molecule/",'coord.xyz')
mol.hess_class = NuclearHessianPM
mol.prepare_prediction()
hessian = mol.predict(model)
mol.trv_properties
print(hessian)
mol.nuclear_properties.calc_properties(shift=False)
trv = TruhlarCramerRRHO(mol)
trv.print_thermoprop()
```
