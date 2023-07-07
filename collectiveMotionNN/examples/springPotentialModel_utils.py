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


def init_SDEwrappers(Module, edgeModule, graph_init, device, noise_type, sde_type, N_batch_edgeUpdate=1, scorePostProcessModule=sm.pAndLogit2KLdiv(), scoreIntegrationModule=sm.scoreListModule()):
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


def run_SDEsimulate(SDEwrapper, x0, t_save, dt_step, device, method_SDE, bm_levy='none'):
    Nd = SDEwrapper.dynamicGNDEmodule.calc_module.N_dim
    peri = SDEwrapper.dynamicGNDEmodule.calc_module.periodic
    
    bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                          size=(x0.shape[0], Nd), dt=dt_step, levy_area_approximation=bm_levy, device=device)

    with torch.no_grad():
        y = sdeint(SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)

    y = y.to('cpu')
    if not(peri is None):
        y[..., :N_dim] = torch.remainder(y[..., :N_dim], peri)

    return y


def run_ODEsimulate(neuralDE, SDEwrapper, graph, x_truth, device, t_learn_span, t_learn_save, useScore=False):
    torch.cuda.empty_cache()
    
    x_truth = x_truth.reshape([-1, x_truth.shape[-1]]).to(device)
    
    if useScore:
        SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
        SDEwrapper.loadGraph(copy.deepcopy(graph).to(device))
        _ = SP_SDEwrapper.f(1, x_truth)
        score_truth = torch.stack(SDEwrapper.score(), dim=1)
        SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(False)
    
    
    SDEwrapper.loadGraph(graph.to(device))
                
    _, x_pred = neuralDE(SDEwrapper.ndataInOutModule.output(SDEwrapper.graph).to(device), 
                         t_learn_span.to(device), save_at=t_learn_save.to(device))
    return x_pred, x_truth


def makeGraphDataLoader(data_path, N_dim, delayPredict, ratio_valid, ratio_test, split_seed=None, batch_size=1, drop_last=False, shuffle=True, pin_memory=True):
    dataset = spm.myDataset(data_path, N_dim=N_dim, delayTruth=delayPredict)
    dataset.initialize()
    
    N_valid = int(dataset.N_batch * ratio_valid)
    N_test = int(dataset.N_batch * ratio_test)
    N_train = dataset.N_batch - N_valid - N_test
    
    range_split = torch.utils.data.random_split(range(dataset.N_batch), [N_train, N_valid, N_test], generator=split_seed)
    
    train_dataset = spm.batchedSubset(dataset, [i for i in range_split[0]])
    valid_dataset = spm.batchedSubset(dataset, [i for i in range_split[1]])
    test_dataset = spm.batchedSubset(dataset, [i for i in range_split[2]])
    
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True)
    if len(test_dataset) > 0:
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


def makeLossFunc(N_dim, useScore, periodic, nondimensionalLoss):
    if nondimensionalLoss:
        lossMakeFunc = spm.myLoss_normalized
    else:
        lossMakeFunc = spm.myLoss
    
    if periodic is None:
        lossFunc = lossMakeFunc(ut.euclidDistance_nonPeriodic(), N_dim=N_dim, useScore=useScore)
    else:
        lossFunc = lossMakeFunc(ut.euclidDistance_periodic(torch.tensor(periodic)), N_dim=N_dim, useScore=useScore)
    return lossFunc


def calcLoss(SDEwrapper, x_pred, x_truth, vLoss_weight, scoreLoss_weight, t_learn_span, device):
    if useScore:
        if len(SDEwrapper.score())==0:
            SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
            _ = SDEwrapper.f(t_learn_span.to(device)[-1], x_pred[0])
            
        score_pred = torch.stack(SDEwrapper.score(), dim=1)
    
        xyloss, vloss, scoreloss = lossFunc(x_pred[0], x_truth, score_pred, score_truth)
        loss = xyloss + vLoss_weight * vloss + scoreLoss_weight * scoreloss
    else:
        xyloss, vloss = lossFunc(x_pred[0], x_truth)
        scoreloss = torch.full([1], torch.nan)
        loss = xyloss + vLoss_weight * vloss
    return loss, xyloss, vloss, scoreloss
