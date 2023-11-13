import os
import sys

import numpy as np
from sklearn.dummy import DummyRegressor

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.io import Output
from hess_ml.src.observables import Observables


class xTBHessTarget:
    def __init__(self) -> None:
        pass

    @checkTiming(enabled=False)
    def ImportTarget(self):
        if os.path.isfile(self.target_file):
            LineList = []

            with open(self.target_file) as fd:
                Lines = [line.rstrip("\n") for line in fd]
                for line in Lines[1:]:
                    LineList += line.split()

            self.target = np.zeros([self.N_atoms * 3, self.N_atoms * 3])

            i = 0

            for j in range(self.N_atoms * 3):
                for k in range(self.N_atoms * 3):
                    self.target[j, k] = float(LineList[i])
                    i += 1
        else:
            self.do_calc = False


class PredictHessian(Output, Observables):
    def gen_hess_from_vec_pred(
        self,
        hess_vec_ab,
        N_atoms,
        R_MI_APF_mat,
        transpose_list,
    ):
        ite_hetero = 0

        Hessian = np.zeros([N_atoms * 3, N_atoms * 3])

        for atom_A in range(N_atoms):
            for atom_B in range(atom_A + 1, N_atoms):
                transpose = False

                if [atom_A, atom_B] in transpose_list:
                    transpose = True

                Hessian = self.fill_matrix_block_AB(
                    hess_vec_ab[ite_hetero],
                    Hessian,
                    R_mat=R_MI_APF_mat,
                    A=atom_A,
                    B=atom_B,
                    transpose=transpose,
                )
                ite_hetero += 1

        for atom_A in range(N_atoms):
            for atom_B in range(N_atoms):
                if atom_A != atom_B:
                    Hessian[
                        3 * atom_A : 3 * atom_A + 3,
                        3 * atom_A : 3 * atom_A + 3,
                    ] -= Hessian[
                        3 * atom_A : 3 * atom_A + 3,
                        3 * atom_B : 3 * atom_B + 3,
                    ]

            Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                + np.transpose(
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3],
                )
            ) / 2

        return Hessian

    @checkTiming(enabled=True)
    def Predict(self, model:DummyRegressor=False):

        """
        Prediction environement. Checks first whether the prediciton shall
        be done and converts the target afterwards into a complet Hessian matrix.
        In addtion writes the hessian into the folder of the predicted species.
        params:
        model:Regressor:
        """

        def damping(x):
            return  1-1/(1+np.exp(-3*(x-6)))


        if not model:
            self.do_calc = False
            print("No model available for the prediction.")
            sys.exit()

        if self.do_calc:

            H_hetero = model.predict(np.array(self.Feature_AB))

            #for i in range(len(H_hetero)):
                #if self.Feature_AB[i][-3] > 4.0:
            #    H_hetero[i] *= damping(self.Feature_AB[i][-3])

            predHess = self.gen_hess_from_vec_pred(
                H_hetero,
                self.N_atoms,
                self.R_MI_APF_mat,
                self.transpose_list,
            )

            self.hessian_to_xtb(os.path.join(self.folder, "MLhessian"), predHess)

            del H_hetero



class ORCAHessTarget:
    def __init__(self) -> None:
        pass

    def ReadTarget(self, file: str) -> None:
        return


class DeltaHessTarget:
    def __init__(self) -> None:
        xTBHessTarget()
        ORCAHessTarget()
