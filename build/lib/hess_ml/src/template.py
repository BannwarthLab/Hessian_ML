from __future__ import annotations

import os
from multiprocessing import Pool, Process
from typing import TYPE_CHECKING

import numpy as np

from hess_ml.src.decorator.decorator import checkTiming
from hess_ml.src.Features.features import FeatureTBlite
from hess_ml.src.processing import TransformPredict, TransformTrain
from hess_ml.src.ReadIn.readin import ReadXYZ
from hess_ml.src.Targets.hessian import ORCAHessTarget, PredictHessian, xTBHessTarget

if TYPE_CHECKING:
    from hess_ml.src.config import GeneralConfig, MoleculeConfig

class TrainMLHessianGFN2xTB(ReadXYZ, FeatureTBlite, xTBHessTarget, TransformTrain):
    def __init__(self) -> None:
        super().__init__()

    def setConfiguration(self, folder: str,general_config:GeneralConfig,molecule_config: MoleculeConfig):
        self.molecule_config = molecule_config
        self.general_config = general_config
        self.folder = folder
        self.xyz_file = os.path.join(self.folder,molecule_config.xyz_file)
        self.target_file = os.path.join(self.folder,molecule_config.target_file)
        self.do_calc = True
        self.solvent = molecule_config.solvent

    def ProcessData(self, model=False):
        self.ImportStructure()
        self.PrintInfo()
        self.ImportFeature()
        self.ImportTarget()
        self.Transform()
        self.Predict(model=model)

    def get_feature(self):
        return self.Feature_AB

    def get_target(self):
        return self.Target_AB

    def PrintInfo(self):
        print(f"Import from {self.folder}")
        print(f"Number of Atoms: {self.N_atoms}")

    @checkTiming(enabled=False)
    def Predict(self, model=False):
        pass


class TestMLHessianGFN2xTB(TransformPredict, PredictHessian, TrainMLHessianGFN2xTB):
    def __init__(self) -> None:
        super().__init__()

    def hessians_difference(self, hess1, hess2):
        self.ImportStructure()

        if self.do_calc:

            self.target_file = os.path.join(self.folder, hess1)
            self.ImportTarget()

            target_file = os.path.join(self.folder, hess2)
            read = xTBHessTarget(target_file, self.N_atoms)
            read.ImportTarget()

            self.hess_diff = self.target - read.target

        else:
            self.hess_diff = np.array([])

    def optimize_step(self):
        print(self.xyz)

        LineList = []
        with open("/home/guests/gfeldmann/projects/hessian/tests/MD342/calc/MD_sim/161/hessian") as fd:
            Lines = [line.rstrip("\n") for line in fd]
            for line in Lines[1:]:
                LineList += line.split()
        fd.close()

        self.target = np.zeros([self.N_atoms * 3, self.N_atoms * 3])

        i = 0

        for j in range(self.N_atoms * 3):
            for k in range(self.N_atoms * 3):
                self.target[j, k] = float(LineList[i])
                i += 1

        self.xyz -= 1e-5*np.matmul(np.linalg.inv(self.target),self.gradient.reshape(-1,1)).reshape(self.xyz.shape)

        print(self.xyz)


class PredictMLHessianxTB(TestMLHessianGFN2xTB):
    def __init__(self) -> None:
        super().__init__()

    def ImportTarget(self):
        pass

class TrainMLHessianORCA(ORCAHessTarget, TrainMLHessianGFN2xTB):
    def __init__(self) -> None:
        super().__init__()

class TrainMLHessianDFT(TrainMLHessianGFN2xTB):
    def __init__(self) -> None:
        super().__init__()


class TestMLHessianORCA(ORCAHessTarget, TestMLHessianGFN2xTB):

    def __init__(self) -> None:
        super().__init__()

class PredictMLHessian(TrainMLHessianORCA):

    def __init__(self) -> None:
        super().__init__()

    def ImportTarget(self):
        pass

