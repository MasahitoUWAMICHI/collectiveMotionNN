import torch
from torch import nn

from collectiveMotionNN.simulator import SDEmodules

class NNSDE_singleStep(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.SDEmodule = SDEmodules.singleBatchSDE()
        
    def forward(self, x):
        

