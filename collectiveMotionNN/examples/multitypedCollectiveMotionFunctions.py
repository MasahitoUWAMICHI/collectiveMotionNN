import numpy as np
import torch
from torch import nn

from scipy import special
from torch.autograd import Function

import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

## prepare functions

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
    

## prepare distance metric

def periodic_distance(x, y, L):
    dr = torch.remainder((x - y), L)
    dr[dr > L/2] = dr[dr > L/2] - L
    return dr

## make module

class multitypedCMsimulate(mo.dynamicGNDEmodule):
    def __init__(self, params):
        super().__init__()

        self.J_chem = J_chemoattractant2D(params['kappa'], params['cutoff'])
        self.J_CF = J_contactFollowing()
        self.J_CIL = J_contactInhibitionOfLocomotion(params['r'])

        self.v0 = nn.Parameter(torch.tensor(params['v0'], requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(params['beta'], requires_grad=True))
        
        self.A_CFs = nn.Parameter(torch.tensor(params['A_CFs'], requires_grad=True))
        self.A_CIL = nn.Parameter(torch.tensor(params['A_CIL'], requires_grad=True))
        self.A_chem = nn.Parameter(torch.tensor(params['A_chem'], requires_grad=True))

        self.A_ext = nn.Parameter(torch.tensor(params['A_ext'], requires_grad=True))

        self.L = params['L']
        self.D = params['D']
        
    def forward(self, edges):
        dx = periodic_distance(edges.dst['x'], edges.src['x'], self.L)

        costheta = torch.cos(edges.dst['theta'])
        sintheta = torch.sin(edges.dst['theta'])

        dx_para = costheta * dx[..., :1] + sintheta * dx[..., 1:]
        dx_perp = costheta * dx[..., 1:] - sintheta * dx[..., :1]

        p_para_src = torch.cos(edges.src['theta'] - edges.dst['theta'])
        p_perp_src = torch.sin(edges.src['theta'] - edges.dst['theta'])

        rot_m_v = self.interactNN(torch.concat((dx_para, dx_perp, 
                                                p_para_src, p_perp_src,
                                                edges.dst['type'], edges.src['type']), -1))

        m_v = torch.concat((costheta * rot_m_v[..., :1] - sintheta * rot_m_v[..., 1:],
                            costheta * rot_m_v[..., 1:] + sintheta * rot_m_v[..., :1]), -1)

        m_theta = self.thetaDotNN(torch.concat((dx_para, dx_perp, 
                                                p_para_src, p_perp_src, 
                                                edges.dst['type'], edges.src['type']), -1))
        
        return {'m': torch.concat((m_v, m_theta), -1)}

    def f(self, t, y, gr, dynamicName, batchIDName):
        return None

    def g(self, t, y, gr, dynamicName, batchIDName):
        return None

