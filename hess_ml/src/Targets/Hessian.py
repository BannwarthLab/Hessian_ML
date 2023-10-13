import numpy as np
import os 
from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.IO import Output
from hess_ml.src.Observables import Observables

class xTBHessTarget:
    def __init__(self) -> None:
        pass

    @checkTiming(enabled=False)
    def ImportTarget(self):
        if os.path.isfile(self.target_file):
            LineList = []

            with open(self.target_file, "r") as fd:
                Lines = [line.rstrip("\n") for line in fd]
                for line in Lines[1:]:
                    LineList += line.split()

            self.target = np.zeros([self.N_atoms * 3, self.N_atoms * 3])

            i = 0

            for k in range(self.N_atoms * 3):
                for l in range(self.N_atoms * 3):
                    self.target[k, l] = float(LineList[i])
                    i += 1
        else:
            self.do_calc = False

        return



class PredictHessian(Output,Observables):

    def gen_hess_from_vec_pred(
            self, hess_vec_ab, N_atoms, R_MI_APF_mat, transpose_list
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
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3
                        ] -= Hessian[
                            3 * atom_A : 3 * atom_A + 3, 3 * atom_B : 3 * atom_B + 3
                        ]

                Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3] = (
                    Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                    + np.transpose(
                        Hessian[3 * atom_A : 3 * atom_A + 3, 3 * atom_A : 3 * atom_A + 3]
                    )
                ) / 2

            return Hessian
    
    @checkTiming(enabled=True)
    def Predict(self,model=False,normalizer=False,selection=False):
        if self.do_calc:

            if selection:  # self.selection
                H_hetero = model.predict(
                    selection.transform(np.array(self.Feature_AB))
                )

            if normalizer:
                H_hetero = model.predict(
                    normalizer.transform(np.array(self.Feature_AB))
                )

                H_hetero = normalizer.inverse_transform(H_hetero)

            else:
                H_hetero = model.predict((np.array(self.Feature_AB)))

            predHess = self.gen_hess_from_vec_pred(
                H_hetero,  self.N_atoms,  self.R_MI_APF_mat, self.transpose_list
            )

            self.hessian_to_xtb(os.path.join(self.folder, "MLhessian"), predHess)

            del H_hetero

        return

class ORCAHessTarget:
    def __init__(self) -> None:
        pass

    def ReadTarget(self, file: str) -> None:
        return

class DeltaHessTarget:
    def __init__(self) -> None:
        xTBHessTarget()
        ORCAHessTarget()

        pass