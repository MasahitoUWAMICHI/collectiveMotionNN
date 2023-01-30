import numpy as np
import torch
from torch import nn

from scipy import special
from torch.autograd import Function

class torch_knFunction(Function):
    @staticmethod
    def forward(ctx, input, n):
        ctx.save_for_backward(input)
        ctx.n = n
        numpy_input = input.cpu().detach().numpy()
        result = special.kn(n, numpy_input)
        return torch.from_numpy(result).to(input.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.cpu().detach().numpy()
        input, = ctx.saved_tensors
        n = ctx.n
        numpy_input = input.cpu().detach().numpy()
        if n==0:
            grad_kn = -special.kn(1, numpy_input)
        else:
            grad_kn = -(special.kn(n-1, numpy_input) + special.kn(n+1, numpy_input))/2
        result = numpy_go * grad_kn
        return torch.from_numpy(result).to(grad_output.device), None

class torch_kn(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def forward(self, input):
        return torch_knFunction.apply(input, self.n)

class torch_kn_cutoff(nn.Module):
    def __init__(self, n, cutoff):
        super().__init__()
        self.n = n
        self.cutoff = cutoff
        
        self.cutoff_val = torch_knFunction.apply(self.cutoff, self.n)
        
        self.cutoff_module = nn.ReLU()
        
    def forward(self, input):
        return self.cutoff_module(torch_knFunction.apply(input, self.n) - self.cutoff_val)

class J_chemoattractant2D(nn.Module):
    def __init__(self, kappa, cutoff):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(kappa, requires_grad=True))
        
        self.cutoff = cutoff
        if self.cutoff > 0:
            self.k1 = torch_kn_cutoff(1, self.cutoff)
        else:
            self.k1 = torch_kn(1)
    
    def forward(self, input):
        return self.k1(self.kappa * input) * (self.kappa/(2*np.pi))
    
class J_contactFollowing(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, xy, p):
        return (1 + xy[...,:1] * p[..., :1] + xy[...,1:2] * p[..., 1:2]) * xy / 2

class J_contactInhibitionOfLocomotion(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(r, requires_grad=True))
    
    def forward(self,xy, d):
        return ((self.r/d) - 1) * xy
    
def periodic_distance(x, y, L):
    dr = torch.remainder((x - y), L)
    dr[dr > L/2] = dr[dr > L/2] - L
    return dr
