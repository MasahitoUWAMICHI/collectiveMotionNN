import numpy as np
import torch
from torch import nn

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
        return self.c * (self.sigma/r)**(self.q) * ((self.sigma/r)**(self.p-self.q) - 1)

    def force(self, r):
        return self.c * (self.sigma/r)**(self.q) * (self.p * (self.sigma/r)**(self.p-self.q) - self.q) / r
    

