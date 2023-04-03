import numpy as np
import torch
from torch import nn

import copy

from torchsde import BrownianInterval, sdeint

import dgl
import dgl.function as fn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

import argparse
from distutils.util import strtobool

import cloudpickle

class interactionModule(nn.Module):
    def __init__(self, u0, w0, sigma=0.1, positionName=None, velocityName=None, polarityName=None, torqueName=None, noiseName=None, messageName=None):
        super().__init__()
        
        self.v0 = v0
        self.w0 = w0

        self.sigma = sigma
        
        self.prepare_sigma()
                    
        self.positionName = ut.variableInitializer(positionName, 'x')
        self.velocityName = ut.variableInitializer(velocityName, 'v')
        self.polarityName = ut.variableInitializer(polarityName, 'theta')
        self.torqueName = ut.variableInitializer(torqueName, 'w')
        self.noiseName = ut.variableInitializer(noiseName, 'sigma')
        
        self.messageName = ut.variableInitializer(messageName, 'm')

        
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([2,1]), self.sigma*torch.ones([1,1])), dim=0)
            
    def calc_message(self, edges):
        dtheta = edges.src[self.positionName] - edges.dst[self.polarityName]
        return {self.messageName: torch.cat((torch.cos(dtheta), torch.sin(dtheta)), -1)}
    
    def aggregate_message(self, nodes):
        mean_cs = torch.mean(nodes.mailbox[self.messageName], 1)
        return {self.torqueName : self.w0 * nn.functional.normalize(mean_cs, dim=-1)[..., 1:2]}
        
    def f(self, t, g, args=None):
        g.ndata[self.velocityName] = self.v0 * torch.cat((torch.cos(g.ndata[self.polarityName]), torch.sin(g.ndata[self.polarityName])), -1)
        g.update_all(self.calc_message, self.aggregate_message)
        return g
      
    def g(self, t, g, args=None):
        g.ndata[self.noiseName] = self.sigmaMatrix.repeat(g.ndata[self.positionName].shape[0], 1, 1).to(g.device)
        return g
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--v0', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--sigma', type=float)
    
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
    parser.add_argument('--save_x_SDE', type=str)
    parser.add_argument('--save_t_SDE', type=str)
    parser.add_argument('--save_model', type=str)

    parser.add_argument('--method_SDE', type=str)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--bm_levy', type=str)
    
    args = parser.parse_args()
    
    
    v0 = ut.variableInitializer(args.v0, 0.03)
    w0 = ut.variableInitializer(args.w0, 1.0)
    
    sigma = ut.variableInitializer(args.sigma, 0.3)
        
    r0 = ut.variableInitializer(args.r0, 1.0)
    L = ut.variableInitializer(args.L, 5.0)
    
    N_particles = ut.variableInitializer(args.N_particles, int(100))
    N_batch = ut.variableInitializer(args.N_batch, int(5))
    
    t_max = ut.variableInitializer(args.t_max, 50.0)
    dt_step = ut.variableInitializer(args.dt_step, 0.1)
    dt_save = ut.variableInitializer(args.dt_save, 1.0)
    
    periodic = ut.variableInitializer(args.periodic, None)
    selfloop = ut.variableInitializer(args.selfloop, False)
    
    device = ut.variableInitializer(args.device, 'cuda' if torch.cuda.is_available() else 'cpu')
    save_x_SDE = ut.variableInitializer(args.save_x_SDE, 'Vicsek_SDE_traj.pt')
    save_t_SDE = ut.variableInitializer(args.save_t_SDE, 'Vicsek_SDE_t_eval.pt')
    save_model = ut.variableInitializer(args.save_model, 'Vicsek_SDE_model.pt')
    
    method_SDE = ut.variableInitializer(args.method_SDE, 'euler')
    noise_type = ut.variableInitializer(args.noise_type, 'general')
    sde_type = ut.variableInitializer(args.sde_type, 'ito')
    
    bm_levy = ut.variableInitializer(args.bm_levy, 'none')
    
    Vicsek_Module = interactionModule(v0, w0, sigma).to(device)
    edgeModule = gu.radiusgraphEdge(r0, periodic, selfloop).to(device)
    
    Vicsek_SDEmodule = mo.dynamicGNDEmodule(Vicsek_Module, edgeModule).to(device)
    
    
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, 2]) * L, (torch.rand([N_particles, 1]) - 0.5) * (2*np.pi)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'theta'], [2, 1])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
        
    
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)

    
    
    
    
    Vicsek_SDEwrapper = mo.dynamicGSDEwrapper(Vicsek_SDEmodule, copy.deepcopy(graph_init).to(device), 
                                          ndataInOutModule=gu.multiVariableNdataInOut(['x', 'theta'], [2, 1]), 
                                          derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'w'], [2, 1]),
                                          noise_type=noise_type, sde_type=sde_type).to(device)
    
    bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                      size=(x0.shape[0], 2), dt=dt_step, device=device)
  
    y = sdeint(Vicsek_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)
    
    print(Vicsek_SDEwrapper.graph)
    
    y = y.to('cpu')
    if not(periodic is None):
        y[..., :2] = torch.remainder(y[..., :2], periodic)
    
    y = y.reshape((t_save.shape[0], N_batch, N_particles, 4))

    torch.save(y, save_x_SDE)

    torch.save(t_save.to('cpu'), save_t_SDE)
    
    with open(save_model, mode='wb') as f:
        cloudpickle.dump(Vicsek_SDEwrapper.to('cpu'), f)
    

    
    
    

    
        
