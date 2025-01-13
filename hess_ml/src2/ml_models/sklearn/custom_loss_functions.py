import torch 
import torch.nn as nn 
import numpy as np 

from hess_ml.src2.utilities.matrix_operation import rotate_matrix
from hess_ml.src2.utilities.geometrical import rot_X

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, output, target):
        loss = 0
        rot_mats = [rot_X(val) for val in np.linspace(0,np.pi,5)]
        N = len(output)
        for R in rot_mats:
            loss += torch.mean(((rotate_matrix(R,output.reshape(N,3,3))-rotate_matrix(R,target.reshape(N,3,3)))**2))
        return loss/3
    
    def forward_non(self, output, target):

        loss = torch.mean(torch.abs(output-target))

        return loss
    
class CustomHuberLoss(nn.Module):
    def __init__(self):
        super(CustomHuberLoss, self).__init__()

    def forward(self, output, target):
        delta = 5e-2
        diff = torch.abs(output-target)
        slope = torch.where(torch.sign(target)==torch.sign(output),1,5)
        loss = torch.where(diff<delta,0.5*diff**2,delta*(diff-delta*0.5))
        loss *= slope
        return loss.sum()
    

class CustomHuberLoss(nn.Module):
    def __init__(self):
        super(CustomHuberLoss, self).__init__()

    def forward(self, output, target):
        delta = 5e-2
        diff = torch.abs(output-target)
        slope = torch.where(torch.sign(target)==torch.sign(output),1,5)
        loss = torch.where(diff<delta,0.5*diff**2,delta*(diff-delta*0.5))
        loss *= slope
        return loss.sum()

# class CustomRelativeError(nn.Module):
#     def __init__(self,delta=5e-2):
#         super(CustomRelativeError, self).__init__()
#         self._delta = delta

#     def forward(self,output,target):
#         diff = torch.abs(target-output)
#         diff = torch.where(diff<self._delta,0.5*(diff)**2,self._delta*(diff-0.5*self._delta))
#         return torch.mean((diff)/(np.abs(target)+5e-2))

class CustomRelativeError(nn.Module):
    def __init__(self,delta=5e-3,relative_delta=1e-4):
        super(CustomRelativeError, self).__init__()
        self._delta = delta
        self._relative_delta = relative_delta

    def forward(self,output,target):
        diff = torch.abs(target-output)
        diff1 = torch.where(diff<self._delta,0.5*(diff)**2,self._delta*(diff-0.5*self._delta))
        delta1 = torch.mean((diff1)/(torch.abs(target)+self._relative_delta))
        diff2 = torch.where(diff<self._delta,0.5*(diff)**2,self._delta*(diff-0.5*self._delta))
        delta2 = torch.mean((diff2))
        return (delta1+delta2)*0.5
