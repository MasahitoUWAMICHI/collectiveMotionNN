import numpy as np
import torch
from torch import nn

import dgl.function as fn

import ..utils as ut
import ..graph_utils as gu
import ..module as mo

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
        return 4 * self.c * (self.sigma/r)**(self.q) * ((self.p * (self.sigma/r)**(self.p-self.q)) - self.q) / r
    

class interactionModule(nn.Module):
    def __init__(self, d, c, sigma, p=12, q=6, periodic=None, dynamicName=None, messageName=None, aggregateName=None):
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
            
        self.dynamicName = ut.variableInitializer(dynamicName, 'y')
        
        self.messageName = ut.variableInitializer(messageName, 'm')

        self.aggregateName = ut.variableInitializer(aggregateName, 'v')
        
    def set_dynamicName(self, dynamicName):
        self.dynamicName = ut.variableInitializer(dynamicName, self.dynamicName)

    def set_messageName(self, messageName):
        self.messageName = ut.variableInitializer(messageName, self.messageName)

    def set_aggregateName(self, aggregateName):
        self.aggregateName = ut.variableInitializer(aggregateName, self.aggregateName)
        
    def calc_message(self, edges):
        dr = calc_dr(edges.dst[self.dynamicName], edges.src[self.dynamicName])

        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {self.messageName: self.LJ.force(abs_dr) * unit_dr}
        
    def f(self, t, g, dynamicName=None, derivativeName=None):
        self.set_dynamicName(dynamicName)
        self.set_aggregateName(derivativeName)
        g.update_all(self.calc_message, fn.sum(self.messageName, self.set_aggregateName))
        return g

    
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
    
    
class edgeCalculator(nn.Module):
    def __init__(self, r0, periodic=None, selfLoop=False):
        super().__init__()
           
        self.r0 = r0

        self.periodic = periodic
        
        self.selfLoop = selfLoop
        
        self.def_dr()
        
        self.def_distance2edge()
        
           
    def def_nonPeriodic(self):
        self.distanceCalc = dr_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = dr_periodic(self.periodic)
        
    def def_dr(self):
        if self.periodic is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic(periodic)
        

    def def_noSelfLoop(self):
        self.distance2edge = gu.distance2edge_noSelfLoop(self.r0)
        
    def def_selfLoop(self):
        self.distance2edge = gu.distance2edge_selfLoop(self.r0)
        
    def def_distance2edge(self):
        if self.selfLoop:
            self.def_selfLoop()
        else:
            self.def_noSelfLoop()
            
            
        
    def forward(self, r):
        dr = self.distanceCalc(torch.unsqueeze(r, 0), torch.unsqueeze(r, 1))
        dr = torch.norm(dr, dim=-1)
        return self.distance2edge(dr)        
    
    
LJ_GODEwrapper(mo.dynamicGNDEmodule):
    def __init__(self, calc_module, edgeConditionFunc, forceUpdate=False)
        super().__init__(calc_module, edgeConditionFunc, forceUpdate)
