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


def init_graph():
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, N_dim]) * L, (torch.rand([N_particles, N_dim])-0.5) * (2*v0)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
    return x0, graph_init

def init_SDEwrapper():
    edgeModule = sm.radiusgraphEdge(r0, periodic, selfloop, multiBatch=N_batch_edgeUpdate>1).to(device)
    
    SP_SDEmodule = wm.dynamicGNDEmodule(SP_Module, edgeModule, returnScore=False, 
                                        scorePostProcessModule=sm.pAndLogit2KLdiv(), scoreIntegrationModule=sm.scoreListModule(),
                                        N_multiBatch=N_batch_edgeUpdate).to(device)

    SP_SDEwrapper = wm.dynamicGSDEwrapper(SP_SDEmodule, copy.deepcopy(graph_init).to(device), 
                                          ndataInOutModule=gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim]), 
                                          derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'a'], [N_dim, N_dim]),
                                          noise_type=noise_type, sde_type=sde_type).to(device)
    return edgeModule, SP_SDEwrapper
