import torch
from torch import nn

def variableInitializer(val, defaultVal):
    if val is None:
        return defaultVal
    else:
        return val

    
class euclidDistance_nonPeriodic(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, r_target, r_zero):
        return r_target - r_zero
    
class euclidDistance_periodic(nn.Module):
    def __init__(self, periodicLength):
        super().__init__()
        
        self.periodicLength = torch.tensor(periodicLength, dtype=torch.float32)
        
    def forward(self, r_target, r_zero):
        dr = torch.remainder(r_target - r_zero, self.periodicLength)
        return dr - (dr > self.periodicLength/2) * self.periodicLength
    
    
