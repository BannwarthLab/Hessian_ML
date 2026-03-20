"A script to compute the ML Hessian for an xyz via CLI"

from mlhess.utils.chemistry.molecule import Molecule
from mlhess.machinelearning.target.hessian import NuclearHessianPM
from mlhess.machinelearning.constructor import load_model
from tcgm_lib.IO.writer import wrt_hess_to_xtb
from tcgm_lib.trv.trv_models.mrrho import TruhlarCramerRRHO

def comp_hessian_from_xyz_cli(fxyz):
    model = load_model()
    mol = Molecule(path="",fxyz=fxyz)
    mol.hess_class = NuclearHessianPM
    mol.prepare_prediction()
    hessian = mol.predict(model)
    wrt_hess_to_xtb("hessian",hessian)
    mol.trv_properties
    mol.nuclear_properties.calc_properties(shift=False)
    trv = TruhlarCramerRRHO(mol)
    trv.print_thermoprop()
