import numpy as np
import torch

import copy

from torchsde import BrownianInterval, sdeint
from torchdyn.core import NeuralODE

import dgl
import dgl.function as fn

from dgl.dataloading import GraphDataLoader

from gradient_descent_the_ultimate_optimizer import gdtuo

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.wrapper_modules as wm
import collectiveMotionNN.sample_modules as sm

import collectiveMotionNN.examples.springPotentialModel as spm

import argparse
from distutils.util import strtobool

import cloudpickle
 

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float)
    parser.add_argument('--r_c', type=float)
    parser.add_argument('--p', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--gamma', type=float)
    
    parser.add_argument('--N_dim', type=int)
    
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
    parser.add_argument('--save_x_SDE', type=str)
    parser.add_argument('--save_t_SDE', type=str)
    parser.add_argument('--save_model', type=str)

    parser.add_argument('--method_SDE', type=str)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--bm_levy', type=str)
    

    parser.add_argument('--skipSimulate', type=strtobool)
    
    parser.add_argument('--c_init', type=float)
    parser.add_argument('--r_c_init', type=float)
    parser.add_argument('--gamma_init', type=float)
    parser.add_argument('--sigma_init', type=float)

    
    parser.add_argument('--delayPredict', type=int)
    parser.add_argument('--dt_train', type=float)
    
    parser.add_argument('--method_ODE', type=str)
    parser.add_argument('--N_epoch', type=int)
    parser.add_argument('--N_train_batch', type=int)
    
    parser.add_argument('--ratio_valid', type=float)
    parser.add_argument('--ratio_test', type=float)
    parser.add_argument('--split_seed', type=int)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_hyperSGD', type=float)
    parser.add_argument('--vLoss_weight', type=float)
    parser.add_argument('--scoreLoss_weight', type=float)
    parser.add_argument('--useScore', type=strtobool)
    
    parser.add_argument('--save_learned_model', type=str)
    parser.add_argument('--save_loss_history', type=str)
    parser.add_argument('--save_validloss_history', type=str)
    return parser

def parser2main(args):
    main(c=args.c, r_c=args.r_c, p=args.p, gamma=args.gamma, sigma=args.sigma, r0=args.r0, L=args.L, v0=args.v0,
         N_dim=args.N_dim, N_particles=args.N_particles, N_batch=args.N_batch, 
         t_max=args.t_max, dt_step=args.dt_step, dt_save=args.dt_save, 
         periodic=args.periodic, selfloop=args.selfloop, 
         device=args.device,
         save_x_SDE=args.save_x_SDE, save_t_SDE=args.save_t_SDE, save_model=args.save_model,
         method_SDE=args.method_SDE, noise_type=args.noise_type, sde_type=args.sde_type, bm_levy=args.bm_levy,
         skipSimulate=args.skipSimulate,
         c_init=args.c_init, r_c_init=args.r_c_init, gamma_init=args.gamma_init, sigma_init=args.sigma_init,
         delayPredict=args.delayPredict, dt_train=args.dt_train, 
         method_ODE=args.method_ODE, 
         N_epoch=args.N_epoch, N_train_batch=args.N_train_batch, 
         ratio_valid=args.ratio_valid, ratio_test=args.ratio_test,
         split_seed=args.split_seed,
         lr=args.lr, lr_hyperSGD=args.lr_hyperSGD, 
         vLoss_weight=args.vLoss_weight, scoreLoss_weight=args.scoreLoss_weight, 
         useScore=args.useScore,
         save_learned_model=args.save_learned_model, 
         save_loss_history=args.save_loss_history, save_validloss_history=args.save_validloss_history)
    
def main(c=None, r_c=None, p=None, gamma=None, sigma=None, r0=None, L=None, v0=None,
         N_dim=None, N_particles=None, N_batch=None, 
         t_max=None, dt_step=None, dt_save=None, 
         periodic=None, selfloop=None, 
         device=None,
         save_x_SDE=None, save_t_SDE=None, save_model=None,
         method_SDE=None, noise_type=None, sde_type=None, bm_levy=None,
         skipSimulate=None,
         c_init=None, r_c_init=None, gamma_init=None, sigma_init=None, 
         delayPredict=None, dt_train=None, 
         method_ODE=None, 
         N_epoch=None, N_train_batch=None, 
         ratio_valid=None, ratio_test=None,
         split_seed=None,
         lr=None, lr_hyperSGD=None, 
         vLoss_weight=None, scoreLoss_weight=None, 
         useScore=None,
         save_learned_model=None, 
         save_loss_history=None, save_validloss_history=None):

    c = ut.variableInitializer(c, 1.0)
    r_c = ut.variableInitializer(r_c, 1.0)
    p = ut.variableInitializer(p, 2.0)
    
    gamma = ut.variableInitializer(gamma, 0.0)
    sigma = ut.variableInitializer(sigma, 0.1)
    
    
    r0 = ut.variableInitializer(r0, 1.0)
    L = ut.variableInitializer(L, 5.0)
    v0 = ut.variableInitializer(v0, 1.0)
    
    N_dim = ut.variableInitializer(N_dim, int(2))
    N_particles = ut.variableInitializer(N_particles, int(100))
    N_batch = ut.variableInitializer(N_batch, int(5))
    
    t_max = ut.variableInitializer(t_max, 50.0)
    dt_step = ut.variableInitializer(dt_step, 0.1)
    dt_save = ut.variableInitializer(dt_save, 1.0)
    
    periodic = ut.variableInitializer(periodic, None)
    selfloop = ut.variableInitializer(selfloop, False)
    
    device = ut.variableInitializer(device, 'cuda' if torch.cuda.is_available() else 'cpu')
    save_x_SDE = ut.variableInitializer(save_x_SDE, 'Spring_SDE_traj.pt')
    save_t_SDE = ut.variableInitializer(save_t_SDE, 'Spring_SDE_t_eval.pt')
    save_model = ut.variableInitializer(save_model, 'Spring_SDE_model.pt')
    
    method_SDE = ut.variableInitializer(method_SDE, 'euler')
    noise_type = ut.variableInitializer(noise_type, 'general')
    sde_type = ut.variableInitializer(sde_type, 'ito')
    
    bm_levy = ut.variableInitializer(bm_levy, 'none')
    

    skipSimulate = ut.variableInitializer(skipSimulate, False)
    
    
    c_init = ut.variableInitializer(c_init, None)
    r_c_init = ut.variableInitializer(r_c_init, None)
    gamma_init = ut.variableInitializer(gamma_init, None)
    
    sigma_init = ut.variableInitializer(sigma_init, None)

    
    delayPredict = ut.variableInitializer(delayPredict, 1)
    dt_train = ut.variableInitializer(dt_train, dt_step)

    method_ODE = ut.variableInitializer(method_ODE, 'euler')
    N_epoch = ut.variableInitializer(N_epoch, 10)
    N_train_batch = ut.variableInitializer(N_train_batch, 8)

    ratio_valid = ut.variableInitializer(ratio_valid, 1.0 / N_batch)
    ratio_test = ut.variableInitializer(ratio_test, 0.0)

    if split_seed is None:
        split_seed = torch.Generator()
    else:
        split_seed = torch.Generator().manual_seed(split_seed)
    
    lr = ut.variableInitializer(lr, 1e-3)
    lr_hyperSGD = ut.variableInitializer(lr_hyperSGD, 1e-3)
    vLoss_weight = ut.variableInitializer(vLoss_weight, 1.0)
    scoreLoss_weight = ut.variableInitializer(scoreLoss_weight, 1.0)
    useScore = ut.variableInitializer(useScore, False)
    
    save_learned_model = ut.variableInitializer(save_learned_model, 'Spring_parametric_learned_model.pt')
    save_loss_history = ut.variableInitializer(save_loss_history, 'Spring_parametric_loss_history.pt')
    save_validloss_history = ut.variableInitializer(save_validloss_history, 'Spring_parametric_validloss_history.pt')
    
  
    
    SP_Module = spm.interactionModule(c, r_c, p, gamma, sigma, N_dim, periodic).to(device)
    edgeModule = sm.radiusgraphEdge(r0, periodic, selfloop).to(device)
    
    SP_SDEmodule = wm.dynamicGNDEmodule(SP_Module, edgeModule, returnScore=False, scorePostProcessModule=sm.pAndLogit2KLdiv(), scoreIntegrationModule=sm.scoreListModule()).to(device)
    
    
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, N_dim]) * L, (torch.rand([N_particles, N_dim])-0.5) * (2*v0)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
        
    
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)

    
    
    
    
    SP_SDEwrapper = wm.dynamicGSDEwrapper(SP_SDEmodule, copy.deepcopy(graph_init).to(device), 
                                          ndataInOutModule=gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim]), 
                                          derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'a'], [N_dim, N_dim]),
                                          noise_type=noise_type, sde_type=sde_type).to(device)
    
    if not skipSimulate:
    
        bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                          size=(x0.shape[0], N_dim), dt=dt_step, levy_area_approximation=bm_levy, device=device)

        with torch.no_grad():
            y = sdeint(SP_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)

        print(SP_SDEwrapper.graph)

        y = y.to('cpu')
        if not(periodic is None):
            y[..., :N_dim] = torch.remainder(y[..., :N_dim], periodic)

        y = y.reshape((t_save.shape[0], N_batch, N_particles, 2*N_dim))

        torch.save(y, save_x_SDE)

        torch.save(t_save.to('cpu'), save_t_SDE)

        with open(save_model, mode='wb') as f:
            cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
    

    
    
    
    SP_SDEwrapper.dynamicGNDEmodule.calc_module.reset_parameter(c_init, r_c_init, gamma_init, sigma_init)
    
    SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_returnScoreMode(useScore)
    
    print(SP_SDEwrapper.state_dict())
    
        
    optim = gdtuo.Adam(alpha=lr, beta1=0.9, beta2=0.999, log_eps=-8., optimizer=gdtuo.SGD(lr_hyperSGD))

    mw = gdtuo.ModuleWrapper(SP_SDEwrapper, optimizer=optim)
    mw.initialize()
    
    
    
    neuralDE = NeuralODE(SP_SDEwrapper, solver=method_ODE).to(device)
    
    
    
    t_pred_max = dt_save * float(delayPredict)
    
    t_learn_span = torch.arange(0, t_pred_max+dt_train, dt_train)
    t_learn_save = torch.tensor([t_pred_max])
    
    
    
    vicsek_dataset = spm.myDataset(save_x_SDE, delayTruth=delayPredict)
    vicsek_dataset.initialize()
    
    N_valid = int(vicsek_dataset.N_batch * ratio_valid)
    N_test = int(vicsek_dataset.N_batch * ratio_test)
    N_train = vicsek_dataset.N_batch - N_valid - N_test
    
    range_split = torch.utils.data.random_split(range(vicsek_dataset.N_batch), [N_train, N_valid, N_test], generator=split_seed)
    
    train_dataset = spm.batchedSubset(vicsek_dataset, [i for i in range_split[0]])
    valid_dataset = spm.batchedSubset(vicsek_dataset, [i for i in range_split[1]])
    test_dataset = spm.batchedSubset(vicsek_dataset, [i for i in range_split[2]])
    
    train_loader = GraphDataLoader(train_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    if len(test_dataset) > 0:
        test_loader = GraphDataLoader(test_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    
    
    if periodic is None:
        lossFunc = spm.myLoss(ut.euclidDistance_nonPeriodic())
    else:
        lossFunc = spm.myLoss(ut.euclidDistance_periodic(torch.tensor(periodic)))
        
    
    
    
    
    
    
    best_valid_loss = np.inf
    
    print('epoch: trainLoss (xy, v, score), validLoss (xy, v, score), c, r_c, gamma, sigma, alpha, 1-beta1, 1-beta2')
    
    loss_history = []
    valid_loss_history = []
        
    for epoch in range(N_epoch):
        for graph, x_truth in train_loader:
            mw.begin()
            graph_batchsize = len(graph.batch_num_nodes())
            
            x_truth = x_truth.reshape([-1, x_truth.shape[-1]]).to(device)
            
            if useScore:
                SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                SP_SDEwrapper.loadGraph(copy.deepcopy(graph).to(device))
                _ = SP_SDEwrapper.f(1, x_truth)
                score_truth = torch.stack(SP_SDEwrapper.score(), dim=1)
                SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(False)
            
            
            SP_SDEwrapper.loadGraph(graph.to(device))
            _, x_pred = neuralDE(SP_SDEwrapper.ndataInOutModule.output(SP_SDEwrapper.graph).to(device), 
                                 t_learn_span.to(device), save_at=t_learn_save.to(device))
            

            if useScore:
                score_pred = torch.stack(SP_SDEwrapper.score(), dim=1)
            
                xyloss, vloss, scoreloss = lossFunc(x_pred[0], x_truth, score_pred, score_truth)
                loss = xyloss + vLoss_weight * vloss + scoreLoss_weight * scoreloss
            else:
                xyloss, thetaloss = lossFunc(x_pred[0], x_truth)
                scoreloss = torch.full([1], torch.nan)
                loss = xyloss + thetaLoss_weight * thetaloss
                
            loss_history.append([xyloss.item(), vloss.item(), scoreloss.item()])
            valid_loss_history.append([np.nan, np.nan, np.nan])
            mw.zero_grad()
            loss.backward()
            for key in mw.optimizer.parameters.keys():
                mw.optimizer.parameters[key].retain_grad()
            mw.step()
            
        mw.begin() # remove graph for autograd
        
        with torch.no_grad():
            valid_loss = 0
            valid_xyloss_total = 0
            valid_vloss_total = 0
            valid_scoreloss_total = 0
            data_count = 0
            
            for graph, x_truth in valid_loader:
                graph_batchsize = len(graph.batch_num_nodes())
                
                x_truth = x_truth.reshape([-1, x_truth.shape[-1]]).to(device)
                
                if useScore:
                    SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                    SP_SDEwrapper.loadGraph(copy.deepcopy(graph).to(device))
                    _ = SP_SDEwrapper.f(1, x_truth)
                    score_truth = torch.stack(SP_SDEwrapper.score(), dim=1)
                    SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(False)

                SP_SDEwrapper.loadGraph(graph.to(device))
                _, x_pred = neuralDE(SP_SDEwrapper.ndataInOutModule.output(SP_SDEwrapper.graph).to(device), 
                                     t_learn_span.to(device), save_at=t_learn_save.to(device))
                
                if useScore:                
                    score_pred = torch.stack(SP_SDEwrapper.score(), dim=1)
                
                    valid_xyloss, valid_thetaloss, valid_scoreloss = lossFunc(x_pred[0], x_truth, score_pred, score_truth)
                    valid_xyloss_total = valid_xyloss_total + valid_xyloss * graph_batchsize
                    valid_thetaloss_total = valid_thetaloss_total + valid_thetaloss * graph_batchsize
                    valid_scoreloss_total = valid_scoreloss_total + valid_scoreloss * graph_batchsize
                    valid_loss = valid_loss + graph_batchsize * (valid_xyloss + thetaLoss_weight * valid_thetaloss + scoreLoss_weight * valid_scoreloss)
                    
                else:
                    valid_xyloss, valid_thetaloss = lossFunc(x_pred[0], x_truth)
                    valid_xyloss_total = valid_xyloss_total + valid_xyloss * graph_batchsize
                    valid_thetaloss_total = valid_thetaloss_total + valid_thetaloss * graph_batchsize
                    valid_scoreloss_total = torch.full([1], torch.nan)
                    valid_loss = valid_loss + graph_batchsize * (valid_xyloss + thetaLoss_weight * valid_thetaloss)
                    
                data_count = data_count + graph_batchsize
                
            valid_loss = valid_loss / data_count
            valid_xyloss_total = valid_xyloss_total / data_count
            valid_vloss_total = valid_vloss_total / data_count
            valid_scoreloss_total = valid_scoreloss_total / data_count
            valid_loss_history[-1] = [valid_xyloss.item(), valid_vloss.item(), valid_scoreloss.item()]
            
            if valid_loss < best_valid_loss:
                SP_SDEwrapper.deleteGraph()
                with open(save_learned_model, mode='wb') as f:
                    cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
                best_valid_loss = valid_loss
                print('{}: {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.2e}, {:.2e}, {:.2e} Best'.format(
                    epoch, loss.item(), xyloss.item(), vloss.item(), scoreloss.item(),
                    valid_loss.item(), valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.c.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.r_c.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.gamma.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sigma.item(),
                    mw.optimizer.parameters['alpha'].item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta1']).item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta2']).item()))
            else:
                print('{}: {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.2e}, {:.2e}, {:.2e}'.format(
                    epoch, loss.item(), xyloss.item(), vloss.item(), scoreloss.item(),
                    valid_loss.item(), valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.c.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.r_c.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.gamma.item(),
                    SP_SDEwrapper.dynamicGNDEmodule.calc_module.sigma.item(),
                    mw.optimizer.parameters['alpha'].item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta1']).item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta2']).item()))
        
    torch.save(torch.tensor(loss_history), save_loss_history)

    torch.save(torch.tensor(valid_loss_history), save_validloss_history)
    
    
    
    
if __name__ == '__main__':

    parser = main_parser()
    
    args = parser.parse_args()
    
    print(args)
    
    parser2main(args)
    
        

