from torch import nn, optim
import torch


class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value = 5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value =  min_value
        
    def forward(self, x):
        return 1. + self.min_value \
            + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1+(x*x)  )))

class ClippedTanh(nn.Module):

    def __init__(self, min_value = 5e-3):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5*(1+0.999*torch.tanh(x))
