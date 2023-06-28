import numpy as np
import torch

import copy

from torchsde import BrownianInterval, sdeint
from torchdyn.core import NeuralODE

import dgl
import dgl.function as fn

from dgl.dataloading import GraphDataLoader

import torch_optimizer as t_opt

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.wrapper_modules as wm
import collectiveMotionNN.sample_modules as sm

import collectiveMotionNN.examples.springPotentialModel as spm


def init_graph(L, v0, N_particles, N_dim, N_batch):
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, N_dim]) * L, (torch.rand([N_particles, N_dim])-0.5) * (2*v0)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
    return x0, graph_init


def init_SDEwrappers(Module, edgeModule, device, noise_type, sde_type, N_batch_edgeUpdate=1, scorePostProcessModule=sm.pAndLogit2KLdiv(), scoreIntegrationModule=sm.scoreListModule()):
    SDEmodule = wm.dynamicGNDEmodule(Module, edgeModule, returnScore=False, 
                                     scorePostProcessModule=scorePostProcessModule, scoreIntegrationModule=scoreIntegrationModule,
                                     N_multiBatch=N_batch_edgeUpdate).to(device)

    SDEwrapper = wm.dynamicGSDEwrapper(SDEmodule, copy.deepcopy(graph_init).to(device), 
                                       ndataInOutModule=gu.multiVariableNdataInOut([Module.positionName, Module.velocityName], 
                                                                                   [Module.N_dim, Module.N_dim]), 
                                       derivativeInOutModule=gu.multiVariableNdataInOut([Module.velocityName, Module.accelerationName], 
                                                                                        [Module.N_dim, Module.N_dim]),
                                       noise_type=noise_type, sde_type=sde_type).to(device)
    return SDEmodule, SDEwrapper


def run_SDEsimulate(SP_SDEwrapper, x0, t_save, dt_step, device, method_SDE, bm_levy='none'):
    Nd = SP_SDEwrapper.dynamicGNDEmodule.calc_module.N_dim
    peri = SP_SDEwrapper.dynamicGNDEmodule.calc_module.periodic
    
    bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                          size=(x0.shape[0], Nd), dt=dt_step, levy_area_approximation=bm_levy, device=device)

    with torch.no_grad():
        y = sdeint(SP_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)

    y = y.to('cpu')
    if not(peri is None):
        y[..., :N_dim] = torch.remainder(y[..., :N_dim], peri)

    return y




