import numpy as np
import torch
from torch import nn
from torchsde import BrownianInterval, sdeint

import dgl
import dgl.function as fn

import collectiveMotionNN.graph.graph_utils as gu
import collectiveMotionNN.simulator.multitypedCollectiveMotionFunctions as mcmf

class multitypedCollectiveMotionSDE(torch.nn.Module):
    def __init__(self, L, periodic, v0, beta, A_CF, A_CIL, r, A, D, noise_type = 'scalar', sde_type = 'ito'):
        super().__init__()
        
        self.noise_type = noise_type
        self.sde_type = sde_type
        
        self.L = L
        self.periodic = periodic
        self.v0 = v0
        self.beta = beta
        self.A_CF = A_CF
        self.A_CIL = A_CIL
        self.r = r
        self.A = A

        self.sigma = torch.zeros((batch_size, state_size, 1), device=device)
        self.sigma[:, 2, 0] = np.sqrt(2*D)
        
        if self.periodic:
            self.custom_distance = lambda x : mcmf.periodic_distance(torch.unsqueeze(x, 0), torch.unsqueeze(x, 1), self.L)
        
        self.J_chem = mcmf.J_chemoattractant2D(kappa, cutoff)
        self.J_CF = mcmf.J_contactFollowing()
        
    # Drift
    def f(self, t, y):
        if self.periodic:
            rg = gu.make_RadiusGraph(y[:, :2], self.r, flg_selfloop=False, flg_custom=True, func_custom_distance=self.custom_distance)
        else:
            rg = gu.make_RadiusGraph(y[:, :2], self.r, flg_selfloop=False)
        
        
        
#        print(y.shape)
        xy = y[:, :2]
#        if self.periodic:
#            xy = xy % L
        xy = xy2distance(xy, self.L)
#        print(np.shape(xy))
#        print(xy[0].shape)
        xy = torch.cat((torch.unsqueeze(xy[0], 2), torch.unsqueeze(xy[1], 2)), 2)
        d = torch.norm(xy, p='fro', dim=2, keepdim=True)   # distance
        dr = torch.heaviside(self.r - d, torch.tensor([0.0], device=device))   # 1 if distance < r, else 0
        xy = dr * torch.nn.functional.normalize(xy, p=2.0, dim=2)   # normalized distance vector
        c = torch.cos(y[:, 2:])
        s = torch.sin(y[:, 2:])

        jcil = J_CIL(xy, d, self.r)
        
        jchem = J_chemMacdonald(xy, d)
        
        dx0 = self.v0 * torch.cat((c, s), 1)
        dx1 = -self.beta * jcil
        dtheta0 = self.A_CF * J_CF(xy, dr, [c,s]) - self.A_CIL * jcil
        dtheta0 = c * dtheta0[:, 1:] - s * dtheta0[:, :1]
        dtheta1 = self.A * c
        dtheta2 = A_Macdonald * (c * jchem[:, 1:] - s * jchem[:, :1])
        return torch.cat((dx0+dx1, dtheta0+dtheta1+dtheta2), 1)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.sigma

