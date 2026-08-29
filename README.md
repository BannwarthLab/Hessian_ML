A python package providing a framework for the training and using machine learning (ML) models to predict Hessian matrices and compute thermodynamic corrections. 

# Installation

`mlhess` requires Python >= 3.11 and is not published on PyPI yet, so it is installed directly from this repository.

```bash
git clone <this-repository-url>
cd hessian_ml
pip install .
```

For development (editable install, plus `pytest`/`ruff`/`mypy`):

```bash
pip install -e .[dev]
```

Notes:
- One dependency, `tcgm_lib`, is installed directly from its git repository (`https://gitlab.git.nrw/rwth-bannwarthlab/open_tcgm_lib_py`), which is publicly accessible.
- Alternatively, `environment.yaml` sets up a conda environment with `tblite`, `dftd4`, and `pytorch` pulled from `conda-forge`/`pytorch` channels, which can be more reliable for these compiled dependencies: `conda env create -f environment.yaml`.
- Installation registers a `mlhess` command-line entry point (see below).

# Basic CLI example

The quickest way to try the package is to predict the Hessian of a single molecule from a `.xyz` file using the bundled default model:

```bash
mlhess --xyz path/to/molecule.xyz
```

This loads the default ML model, computes the Hessian, writes it to a `hessian` file (xtb format) in the current directory, and prints thermodynamic (RRHO) properties to stdout. Runtime depends on molecule size and thread count, since the Hessian is built from numerical single-point evaluations; set `OMP_NUM_THREADS` to use more than the default single thread, e.g. `OMP_NUM_THREADS=4 mlhess --xyz path/to/molecule.xyz`.

Other CLI options:

```bash
mlhess --help                  # show all options
mlhess --input config.toml     # run the full collect/train/predict pipeline from a .toml config
```

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
