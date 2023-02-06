import torch
from torch import nn

class torchSDEmodule(nn.Module):
    def __init__(self, module_f, module_g):
        super().__init__()
        
        self.module_f = module_f
        
        self.module_g = module_g
        
    def f(self, t, y):
        return self.module_f(y)
    
    def g(self, t, y):
        return self.module_g(y)

        
        

