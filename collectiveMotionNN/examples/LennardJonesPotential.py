import numpy as np
import torch
from torch import nn

import dgl.function as fn

import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

class LJpotential(nn.Module):
    def __init__(self, c, sigma, p=12, q=6):
        super().__init__()
        self.c = c
        self.sigma = sigma
        self.p = p
        self.q = q
        
    def potential(self, r):
        return 4 * self.c * (self.sigma/r)**(self.q) * ((self.sigma/r)**(self.p-self.q) - 1)

    def force(self, r):
        return 4 * self.c * (self.sigma/r)**(self.q) * (self.p * (self.sigma/r)**(self.p-self.q) - self.q) / r
    

class MessagePasser(nn.Module):
    def __init__(self, d, c, sigma, p=12, q=6, periodic=None):
        self.d = d
        self.c = c
        self.sigma = sigma
        self.p = p
        self.q = q
        
        self.LJ = LJpotential(c, sigma, p, q)
        
        self.flg_periodic = not(periodic is None)
        
        if self.flg_periodic:
            self.periodic = torch.tensor(periodic)
        else:
            self.periodic = periodic
        
    def calc_dr(self, r1, r2):
        dr = r2 - r1
        if self.flg_periodic:
            dr = torch.remainder(dr, self.periodic)
            dr[dr > self.periodic/2] = dr[dr > self.periodic/2] - self.periodic
        return dr
        
    def calc_message(self, edges):
        dr = calc_dr(edges.dst['x'], edges.src['x'])

        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {'m': self.LJ(abs_dr) * unit_dr}
        
    def forward(self, g):
        g.update_all(self.calc_message, fn.sum('m', 'v'))
        return g

    
dynamicGODEwrapper(nn.Module):
    def __init__(self, module_f)    
