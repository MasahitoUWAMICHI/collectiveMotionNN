import numpy as np
import torch
from torch import nn

import dgl.nn as gnn

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
    

class MessagePasser(gnn):
    


    
dynamicGODEwrapper(nn.Module):
    def __init__(self, module_f)    
