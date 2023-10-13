import os
import sys
from hess_ml.src.ReadIn.readin import ReadXYZ
from hess_ml.src.Features.Features import FeatureTBlite
from hess_ml.src.Targets.Hessian import xTBHessTarget,PredictHessian
from hess_ml.src.Processing import TransformTrain, TransformPredict
from hess_ml.src.decorator.decorator import checkTiming
import time as time 


class TrainMLHessianGFN2xTB(ReadXYZ,FeatureTBlite,xTBHessTarget,TransformTrain):

    def __init__(self) -> None:
        super().__init__()

    def setConfiguration(self, folder:str, config:dict):
        self.config = config
        self.folder = folder
        self.xyz_file = os.path.join(self.folder,config.get("xyz_file", "xtbopt.xyz"))
        self.gradient_file = os.path.join(self.folder,config.get("gradient_file", "gradient"))
        self.target_file = os.path.join(self.folder,config.get("hessian_file", "hessian"))
        self.do_calc = True
        return

    def ProcessData(self,model=False,normalizer=False,selection=False):
        self.ImportStructure()
        self.PrintInfo()
        self.ImportFeature()
        self.ImportTarget()
        self.Transform()
        self.Predict(model=model,normalizer=normalizer,selection=selection)

    def PrintInfo(self):
        print(f'Import from {self.folder}')
        print(f'Number of Atoms: {self.N_atoms}')
        return

    @checkTiming(enabled=False)
    def Predict(self,model=False,normalizer=False,selection=False):
        pass 


class TestMLHessianGFN2xTB(TransformPredict,PredictHessian,TrainMLHessianGFN2xTB):

    def __init__(self) -> None:
        super().__init__()

    def hessians_difference(self, hess1, hess2):
        self.ImportStructure()

        self.target_file = os.path.join(self.folder, hess1)
        self.ImportTarget()
        self.hessian1 = self.target

        self.target_file = os.path.join(self.folder, hess2)
        self.ImportTarget()

        self.hess_diff = self.hessian1 - self.target

        return

class PredictMLHessian(TestMLHessianGFN2xTB):

    def __init__(self) -> None:
        super().__init__()

    def ImportTarget(self):
        pass 
    

class TrainMLHessianDFT(TrainMLHessianGFN2xTB):

    def __init__(self) -> None:
        super().__init__()

    def ImportTarget(self):
        pass 


class PredictMLHessian(TrainMLHessianDFT):

    def __init__(self) -> None:
        super().__init__()

    def ImportTarget(self):
        pass 
    

