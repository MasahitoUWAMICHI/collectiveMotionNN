from torch import nn


from torchsde import BrownianInterval, sdeint

class torchODEmodule(nn.Module):
    def __init__(self, module_f):
        super().__init__()
                
        self.module_f = module_f
                
    def f(self, t, y):
        return self.module_f(y)


class torchSDEmodule(nn.Module):
    def __init__(self, module_f, module_g, noise_type, sde_type):
        super().__init__()
                
        self.noise_type = noise_type
        self.sde_type = sde_type
        
        self.module_f = module_f
        
        self.module_g = module_g
        
    def f(self, t, y):
        return self.module_f(y)
    
    def g(self, t, y):
        return self.module_g(y)

        

