import torch
from torch import nn


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        delta = 5e-2
        diff = torch.abs(output - target)
        slope = torch.where(torch.sign(target) == torch.sign(output), 1, 5)
        loss = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - delta * 0.5))
        loss *= slope
        return loss.sum()


class RelHuberLoss(nn.Module):
    def __init__(self, delta=5e-3, relative_delta=1e-4):
        super().__init__()
        self._delta = delta
        self._relative_delta = relative_delta

    def forward(self, output, target):
        diff = torch.abs(target - output)
        diff1 = torch.where(
            diff < self._delta,
            0.5 * (diff) ** 2,
            self._delta * (diff - 0.5 * self._delta),
        )
        delta1 = torch.mean((diff1) / (torch.abs(target) + self._relative_delta))
        diff2 = torch.where(
            diff < self._delta,
            0.5 * (diff) ** 2,
            self._delta * (diff - 0.5 * self._delta),
        )
        delta2 = torch.mean(diff2)
        return (delta1 + delta2) * 0.5
