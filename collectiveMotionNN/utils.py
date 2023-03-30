import torch
from torch import nn

def variableInitializer(val, defaultVal):
    if val is None:
        return defaultVal
    else:
        return val

    
class dr_nonPeriodic(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, r1, r2):
        return r2 - r1
    
class dr_periodic(nn.Module):
    def __init__(self, periodic):
        super().__init__()
        
        self.periodic = torch.tensor(periodic, dtype=torch.float32)
        
    def forward(self, r1, r2):
        dr = torch.remainder(r2 - r1, self.periodic)
        return dr - (dr > self.periodic/2) * self.periodic
    
    
