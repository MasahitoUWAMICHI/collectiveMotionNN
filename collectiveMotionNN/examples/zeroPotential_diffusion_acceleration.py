import numpy as np
import torch
from torch import nn

import copy

from torchdyn.core import NeuralODE

from torchsde import BrownianInterval, sdeint

import dgl
import dgl.function as fn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

import argparse
from distutils.util import strtobool

import cloudpickle
    
class zeroPotential(nn.Module):
    def __init__(self):
        super().__init__()
        
    def potential(self, r):
        return r * 0

    def force(self, r):
        return r * 0
    

class interactionModule(nn.Module):
    def __init__(self, gamma=0.0, sigma=0.1, periodic=None, positionName=None, velocityName=None, accelerationName=None, noiseName=None, messageName=None):
        super().__init__()
        
        self.gamma = gamma
        
        self.sigma = sigma
        
        self.prepare_sigma()
        
        self.zp = zeroPotential()
        
        self.flg_periodic = not(periodic is None)
        
        if self.flg_periodic:
            self.periodic = torch.tensor(periodic)
        else:
            self.periodic = periodic
            
        self.def_dr()
            
        self.positionName = ut.variableInitializer(positionName, 'x')
        self.velocityName = ut.variableInitializer(velocityName, 'v')        
        self.accelerationName = ut.variableInitializer(accelerationName, 'a')
        self.noiseName = ut.variableInitializer(noiseName, 'sigma')

        
        self.messageName = ut.variableInitializer(messageName, 'm')

        
    def def_nonPeriodic(self):
        self.distanceCalc = ut.euclidDistance_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = ut.euclidDistance_periodic(self.periodic)
        
    def def_dr(self):
        if self.periodic is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic()
            
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([2,2]), self.sigma*torch.eye(2)), dim=0)
            
    def calc_message(self, edges):
        dr = self.distanceCalc(edges.dst[self.positionName], edges.src[self.positionName])

        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {self.messageName: self.zp.force(abs_dr) * unit_dr}
        
    def f(self, t, g, args=None):
        g.update_all(self.calc_message, fn.sum(self.messageName, self.accelerationName))
        g.ndata[self.accelerationName] = g.ndata[self.accelerationName] - self.gamma * g.ndata[self.velocityName]
        return g
      
    def g(self, t, g, args=None):
        g.ndata[self.noiseName] = self.sigmaMatrix.repeat(g.ndata[self.velocityName].shape[0], 1, 1).to(g.device)
        return g
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--sigma', type=float)
    
    parser.add_argument('--r0', type=float)
    
    parser.add_argument('--L', type=float)
    parser.add_argument('--v0', type=float)
    
    parser.add_argument('--N_particles', type=int)
    parser.add_argument('--N_batch', type=int)

    parser.add_argument('--t_max', type=float)
    parser.add_argument('--dt_step', type=float)
    parser.add_argument('--dt_save', type=float)

    parser.add_argument('--periodic', type=float)
    parser.add_argument('--selfloop', type=strtobool)
    
    parser.add_argument('--device', type=str)
    parser.add_argument('--save_x_ODE', type=str)
    parser.add_argument('--save_t_ODE', type=str)
    parser.add_argument('--save_x_SDE', type=str)
    parser.add_argument('--save_t_SDE', type=str)
    parser.add_argument('--save_model', type=str)
    parser.add_argument('--save_bm', type=str)
    
    parser.add_argument('--method_ODE', type=str)
    parser.add_argument('--method_SDE', type=str)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--bm_levy', type=str)
    
    args = parser.parse_args()
    
        
    gamma = ut.variableInitializer(args.gamma, 0.0)
    sigma = ut.variableInitializer(args.sigma, 0.1)
    
    
    r0 = ut.variableInitializer(args.r0, 3.0)
    L = ut.variableInitializer(args.L, 5.0)
    v0 = ut.variableInitializer(args.v0, 1.0)
    
    N_particles = ut.variableInitializer(args.N_particles, int(10))
    N_batch = ut.variableInitializer(args.N_batch, int(5))
    
    t_max = ut.variableInitializer(args.t_max, 50.0)
    dt_step = ut.variableInitializer(args.dt_step, 0.1)
    dt_save = ut.variableInitializer(args.dt_save, 1.0)
    
    periodic = ut.variableInitializer(args.periodic, None)
    selfloop = ut.variableInitializer(args.selfloop, False)
    
    device = ut.variableInitializer(args.device, 'cuda' if torch.cuda.is_available() else 'cpu')
    save_x_ODE = ut.variableInitializer(args.save_x_ODE, 'ZPacc_ODE_traj.pt')
    save_t_ODE = ut.variableInitializer(args.save_t_ODE, 'ZPacc_ODE_t_eval.pt')
    save_x_SDE = ut.variableInitializer(args.save_x_SDE, 'ZPacc_SDE_traj.pt')
    save_t_SDE = ut.variableInitializer(args.save_t_SDE, 'ZPacc_SDE_t_eval.pt')
    save_model = ut.variableInitializer(args.save_model, 'ZPacc_SDE_model.pt')

    save_bm = ut.variableInitializer(args.save_bm, 'ZPacc_SDE_bm.pt')
    
    method_ODE = ut.variableInitializer(args.method_ODE, 'euler')
    method_SDE = ut.variableInitializer(args.method_SDE, 'euler')
    noise_type = ut.variableInitializer(args.noise_type, 'general')
    sde_type = ut.variableInitializer(args.sde_type, 'ito')
    
    bm_levy = ut.variableInitializer(args.bm_levy, 'None')
    
    SP_Module = interactionModule(gamma, sigma, periodic).to(device)
    edgeModule = gu.radiusgraphEdge(r0, periodic, selfloop).to(device)
    
    SP_SDEmodule = mo.dynamicGNDEmodule(SP_Module, edgeModule).to(device)
    
    
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, 2]) * L, (torch.rand([N_particles, 2]) - 0.5) * (2*v0)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'v'], [2, 2])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
        
    
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)

    
    
    
    
    SP_SDEwrapper = mo.dynamicGSDEwrapper(SP_SDEmodule, copy.deepcopy(graph_init).to(device), 
                                          ndataInOutModule=gu.multiVariableNdataInOut(['x', 'v'], [2, 2]), 
                                          derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'a'], [2, 2]),
                                          noise_type=noise_type, sde_type=sde_type).to(device)
    
    bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                      size=(x0.shape[0], 2), dt=dt_step/4.0, levy_area_approximation=bm_levy, device=device)
  
    y = sdeint(SP_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)
    
    print(SP_SDEwrapper.graph)
    
    if periodic is None:
        y = y.to('cpu')
    else:
        y = torch.remainder(y.to('cpu'), periodic) 
    
    y = y.reshape((t_save.shape[0], N_batch, N_particles, 4))

    torch.save(y, save_x_SDE)

    torch.save(t_save.to('cpu'), save_t_SDE)
    
    #torch.save(SP_SDEwrapper.to('cpu'), save_model)
    with open(save_model, mode='wb') as f:
        cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
    
    with open(save_bm, mode='wb') as f:
        cloudpickle.dump(bm.to('cpu'), f)
        
        
    #SP_SDEwrapper = mo.dynamicGSDEwrapper(SP_SDEmodule, copy.deepcopy(graph_init).to(device), 
    #                                      ndataInOutModule=gu.multiVariableNdataInOut(['x', 'v'], [2, 2]), 
    #                                      derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'a'], [2, 2]),
    #                                      noise_type=noise_type, sde_type=sde_type).to(device)
    
    SP_SDEwrapper.graph =  copy.deepcopy(graph_init).to(device)
    SP_SDEwrapper.edgeInitialize()
    
    neuralDE = NeuralODE(SP_SDEwrapper, solver=method_ODE).to(device)
    
    t_eval, x = neuralDE(x0.to(device), t_span.to(device), save_at=t_save.to(device))
    
    print(neuralDE.vf.vf.graph)
    
    if periodic is None:
        x = x.to('cpu')
    else:
        x = torch.remainder(x.to('cpu'), periodic) 
    
    x = x.reshape((t_eval.shape[0], N_batch, N_particles, 4))

    torch.save(x, save_x_ODE)

    torch.save(t_eval.to('cpu'), save_t_ODE)
    
    
    

    
        
