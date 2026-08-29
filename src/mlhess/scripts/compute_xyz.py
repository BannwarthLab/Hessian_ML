"A script to compute the ML Hessian for an xyz via CLI"

from tcgm_lib.IO.writer import wrt_hess_to_xtb
from tcgm_lib.trv.trv_models.mrrho import TruhlarCramerRRHO

from mlhess.machinelearning.constructor import load_model
from mlhess.machinelearning.target.hessian import NuclearHessianPM
from mlhess.utils.chemistry.molecule import Molecule


def comp_hessian_from_xyz_cli(fxyz):
    model = load_model()
    mol = Molecule(path="",fxyz=fxyz,symmetry=(None,None))
    mol.hess_class = NuclearHessianPM
    mol.prepare_prediction()
    hessian = mol.predict(model)
    wrt_hess_to_xtb("hessian",hessian)
    mol.trv_properties #noQA :B018 
    mol.nuclear_properties.calc_properties(shift=False)
    trv = TruhlarCramerRRHO(mol)
    trv.print_thermoprop()
