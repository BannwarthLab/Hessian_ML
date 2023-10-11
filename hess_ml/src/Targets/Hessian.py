import numpy as np
import os 

class xTBHessTarget:
    def __init__(self) -> None:
        pass


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



class PredictTarget:
    def __init__(self) -> None:
        pass

    def ReadTarget(self, file: str) -> None:

        return

class ORCAHessTarget:
    def __init__(self) -> None:
        pass

    def ReadTarget(self, file: str) -> None:
        return

class DeltaHessTarget:
    def __init__(self) -> None:
        Hessian_xTB = xTBHessTarget()
        Hessian_ORCA = ORCAHessTarget()

        pass