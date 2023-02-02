import torch
from torch import nn

class singleBatchSDE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # module F and G should be overwritten
        self.module_f = lambda x: x
        
        self.module_g = lambda x: torch.zeros_like(x)
        
    def f(self, t, y):
        return self.module_f(y)
    
    def g(self, t, y):
        return self.module_g(y)
    
    
class multiBatchSDE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.module_f = lambda x: x
        
        self.module_g = lambda x: torch.zeros_like(x)
        
    def f(self, t, y):
        return self.module_f(y)
    
    def g(self, t, y):
        return self.module_g(y)
        
        
        

