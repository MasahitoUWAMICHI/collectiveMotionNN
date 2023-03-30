import numpy as np
import torch
from torch import nn

from torchdyn.core import NeuralODE

import dgl
import dgl.function as fn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

import argparse
from distutils.util import strtobool


    
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
    def __init__(self, r0, periodic=None, selfLoop=False, variableName=None):
        super().__init__()
           
        self.r0 = r0

        self.periodic = periodic
        
        self.selfLoop = selfLoop
        
        self.edgeVariable = ut.variableInitializer(variableName, 'x')
        
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
            self.def_periodic()
        

    def def_noSelfLoop(self):
        self.distance2edge = gu.distance2edge_noSelfLoop(self.r0)
        
    def def_selfLoop(self):
        self.distance2edge = gu.distance2edge_selfLoop(self.r0)
        
    def def_distance2edge(self):
        if self.selfLoop:
            self.def_selfLoop()
        else:
            self.def_noSelfLoop()
            
            
        
    def forward(self, g, args=None):
        dr = self.distanceCalc(torch.unsqueeze(g.ndata[self.edgeVariable], 0), torch.unsqueeze(g.ndata[self.edgeVariable], 1))
        dr = torch.norm(dr, dim=-1, keepdim=False)
        return self.distance2edge(dr)        
    

    
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
    def __init__(self, c, sigma, p=12, q=6, periodic=None, dynamicName=None, messageName=None, aggregateName=None):
        super().__init__()
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
            
        self.def_dr()
            
        self.dynamicName = ut.variableInitializer(dynamicName, 'x')
        
        self.messageName = ut.variableInitializer(messageName, 'm')

        self.aggregateName = ut.variableInitializer(aggregateName, 'v')
        
    def set_dynamicName(self, dynamicName):
        self.dynamicName = ut.variableInitializer(dynamicName, self.dynamicName)

    def set_messageName(self, messageName):
        self.messageName = ut.variableInitializer(messageName, self.messageName)

    def set_aggregateName(self, aggregateName):
        self.aggregateName = ut.variableInitializer(aggregateName, self.aggregateName)
        
        
        
    def def_nonPeriodic(self):
        self.distanceCalc = dr_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = dr_periodic(self.periodic)
        
    def def_dr(self):
        if self.periodic is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic()
            
            
    def calc_message(self, edges):
        dr = self.distanceCalc(edges.dst[self.dynamicName], edges.src[self.dynamicName])

        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {self.messageName: self.LJ.force(abs_dr) * unit_dr}
        
    def f(self, t, g, args=None):
        g.update_all(self.calc_message, fn.sum(self.messageName, self.aggregateName))
        return g
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--p', type=float)
    parser.add_argument('--q', type=float)
    
    parser.add_argument('--r0', type=float)
    
    parser.add_argument('--L', type=float)
    
    parser.add_argument('--N_particles', type=int)
    parser.add_argument('--N_batch', type=int)

    parser.add_argument('--t_max', type=float)
    parser.add_argument('--dt_step', type=float)
    parser.add_argument('--dt_save', type=float)

    parser.add_argument('--periodic', type=float)
    parser.add_argument('--selfloop', type=strtobool)
    
    parser.add_argument('--device', type=str)
    parser.add_argument('--save_x', type=str)
    parser.add_argument('--save_t', type=str)
    
    args = parser.parse_args()
    
    
    c = ut.variableInitializer(args.c, 1.0)
    sigma = ut.variableInitializer(args.c, 1.0)
    p = ut.variableInitializer(args.p, 12.0)
    q = ut.variableInitializer(args.q, 6.0)
    
    r0 = ut.variableInitializer(args.r0, 3.0)
    L = ut.variableInitializer(args.L, 5.0)
    
    N_particles = ut.variableInitializer(args.N_particles, int(10))
    N_batch = ut.variableInitializer(args.N_batch, int(5))
    
    t_max = ut.variableInitializer(args.t_max, 50.0)
    dt_step = ut.variableInitializer(args.dt_step, 0.1)
    dt_save = ut.variableInitializer(args.dt_save, 1.0)
    
    periodic = ut.variableInitializer(args.periodic, None)
    selfloop = ut.variableInitializer(args.selfloop, False)
    
    device = ut.variableInitializer(args.device, 'cuda' if torch.cuda.is_available() else 'cpu')
    save_x = ut.variableInitializer(args.save_x, 'LJ_traj.pt')
    save_t = ut.variableInitializer(args.save_t, 't_eval.pt')
    
    
    
    LJ_Module = interactionModule(c, sigma, p, q, periodic)
    edgeModule = edgeCalculator(r0, periodic, selfloop)
    
    LJ_ODEmodule = mo.dynamicGNDEmodule(LJ_Module, edgeModule)
    
    
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.rand([N_particles, 2]) * L)
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.singleVariableNdataInOut('x')))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init).to(device)
        
    
                 
    LJ_ODEwrapper = mo.dynamicGODEwrapper(LJ_ODEmodule, graph_init).to(device)
    
    neuralDE = NeuralODE(LJ_ODEwrapper, solver='euler').to(device)
    
    
    
    
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)
    
    t_eval, x = neuralDE(x0.to(device), t_span.to(device), save_at=t_save.to(device))
    
    print(neuralDE.vf.vf.graph)
    
    if periodic is None:
        x = x.to('cpu')
    else:
        x = torch.remainder(x.to('cpu'), periodic) 
    
    x = x.reshape((t_eval.shape[0], N_batch, N_particles, 2))

    torch.save(x, save_x)

    torch.save(t_eval.to('cpu'), save_t)
