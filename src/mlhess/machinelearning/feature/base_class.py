import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Feature:
    def __init__(self, mol) -> None:
        self._mol = mol
        self._processed: np.ndarray | list | None = None
        self.new_keys: None | list = None

        # atomwise
        self._scalars :np.ndarray |None = None
        self._vectors :np.ndarray |None= None
        self._matrices :np.ndarray |None= None

        # atomwise
        self.scalar_keys = None
        self.vectors_keys = None
        self.matrices_keys = None

        # atompairwise
        self._distance_matrix = None
        self._C6params = None
        self.wbo = None
        self._dipm_key = None
        return

    @property
    def processed(self):
        return self._processed

    @processed.setter
    def processed(self, vals):
        self._processed = vals

    @property
    def scalars(self)->np.ndarray|None:
        return self._scalars

    @scalars.setter
    def scalars(self, vals:np.ndarray):
        self._scalars = vals

    @property
    def vectors(self)->np.ndarray|None:
        return self._vectors

    @vectors.setter
    def vectors(self, vals:np.ndarray):
        self._vectors = vals

    @property
    def matrices(self)->np.ndarray|None:
        return self._matrices

    @matrices.setter
    def matrices(self, vals:np.ndarray): 
        self._matrices = vals

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, vals):
        self._distance_matrix = vals

    @property
    def C6_params(self):
        return self._C6params

    @C6_params.setter  # type: ignore[attr-defined]
    def C6params(self, vals):
        self._C6params = vals

    @property
    def dipm_key(self):
        if self._dipm_key is None:
            for idx, key in enumerate(self.scalar_keys):
                if key == "dipm_A":
                    break

            if key != "dipm_A":
                print("dipm_A not found.")

            self._dipm_key = idx

        return self._dipm_key
