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

import collectiveMotionNN.examples.continuousTimeVicsek as ctv

import argparse
from distutils.util import strtobool

import cloudpickle

import time
import os
 

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float)
    parser.add_argument('--d', type=float)
    parser.add_argument('--u0', type=float)
    parser.add_argument('--sigma', type=float)
    
    parser.add_argument('--N_dim', type=int)
    
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
    
    parser.add_argument('--save_directory_simulation', type=str)
    parser.add_argument('--save_x_SDE', type=str)
    parser.add_argument('--save_t_SDE', type=str)
    parser.add_argument('--save_model', type=str)

    parser.add_argument('--method_SDE', type=str)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--bm_levy', type=str)
    

    parser.add_argument('--skipSimulate', type=strtobool)
    
    parser.add_argument('--c_init', type=float)
    parser.add_argument('--u0_init', type=float)
    parser.add_argument('--sigma_init', type=float)

    
    parser.add_argument('--delayPredict', type=int)
    parser.add_argument('--dt_train', type=float)
    
    parser.add_argument('--method_ODE', type=str)
    parser.add_argument('--N_epoch', type=int)
    parser.add_argument('--N_train_batch', type=int)
    parser.add_argument('--N_batch_edgeUpdate', type=int)
    
    parser.add_argument('--ratio_valid', type=float)
    parser.add_argument('--ratio_test', type=float)
    parser.add_argument('--split_seed_val', type=int)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_hyperSGD', type=float)
    parser.add_argument('--vLoss_weight', type=float)
    parser.add_argument('--scoreLoss_weight', type=float)
    parser.add_argument('--useScore', type=strtobool)
    
    parser.add_argument('--save_directory_learning', type=str)
    parser.add_argument('--save_learned_model', type=str)
    parser.add_argument('--save_loss_history', type=str)
    parser.add_argument('--save_validloss_history', type=str)
    parser.add_argument('--save_run_time_history', type=str)
    parser.add_argument('--save_params', type=str)
    
    return parser

def parser2main(args):
    main(c=args.c, d=args.d, u0=args.u0, sigma=args.sigma, r0=args.r0, L=args.L,
         N_dim=args.N_dim, N_particles=args.N_particles, N_batch=args.N_batch, 
         t_max=args.t_max, dt_step=args.dt_step, dt_save=args.dt_save, 
         periodic=args.periodic, selfloop=args.selfloop, 
         device=args.device,
         save_directory_simulation=args.save_directory_simulation,
         save_x_SDE=args.save_x_SDE, save_t_SDE=args.save_t_SDE, save_model=args.save_model,
         method_SDE=args.method_SDE, noise_type=args.noise_type, sde_type=args.sde_type, bm_levy=args.bm_levy,
         skipSimulate=args.skipSimulate,
         c_init=args.c_init, u0_init=args.u0_init, sigma_init=args.sigma_init,
         delayPredict=args.delayPredict, dt_train=args.dt_train, 
         method_ODE=args.method_ODE, 
         N_epoch=args.N_epoch, N_train_batch=args.N_train_batch, N_batch_edgeUpdate=args.N_batch_edgeUpdate,
         ratio_valid=args.ratio_valid, ratio_test=args.ratio_test,
         split_seed_val=args.split_seed_val,
         lr=args.lr, lr_hyperSGD=args.lr_hyperSGD, 
         vLoss_weight=args.vLoss_weight, scoreLoss_weight=args.scoreLoss_weight, 
         useScore=args.useScore,
         save_directory_learning=args.save_directory_learning,
         save_learned_model=args.save_learned_model, 
         save_loss_history=args.save_loss_history, save_validloss_history=args.save_validloss_history,
         save_run_time_history=args.save_run_time_history,
         save_params=args.save_params)
    
def main(c=None, d=None, u0=None, sigma=None, r0=None, L=None,
         N_dim=None, N_particles=None, N_batch=None, 
         t_max=None, dt_step=None, dt_save=None, 
         periodic=None, selfloop=None, 
         device=None,
         save_directory_simulation=None,
         save_x_SDE=None, save_t_SDE=None, save_model=None,
         method_SDE=None, noise_type=None, sde_type=None, bm_levy=None,
         skipSimulate=None,
         c_init=None, u0_init=None, sigma_init=None, 
         delayPredict=None, dt_train=None, 
         method_ODE=None, 
         N_epoch=None, N_train_batch=None, N_batch_edgeUpdate=None,
         ratio_valid=None, ratio_test=None,
         split_seed_val=None,
         lr=None, lr_hyperSGD=None, 
         vLoss_weight=None, scoreLoss_weight=None, 
         useScore=None,
         save_directory_learning=None,
         save_learned_model=None, 
         save_loss_history=None, save_validloss_history=None,
         save_run_time_history=None,
         save_params=None):

    c = ut.variableInitializer(c, 1.0)
    d = ut.variableInitializer(d, 1.0)
    u0 = ut.variableInitializer(u0, 0.03)
    
    sigma = ut.variableInitializer(sigma, 0.3)
    
    
    r0 = ut.variableInitializer(r0, 1.0)
    L = ut.variableInitializer(L, 5.0)
    
    N_dim = ut.variableInitializer(N_dim, int(2))
    N_particles = ut.variableInitializer(N_particles, int(100))
    N_batch = ut.variableInitializer(N_batch, int(5))
    
    t_max = ut.variableInitializer(t_max, 50.0)
    dt_step = ut.variableInitializer(dt_step, 0.1)
    dt_save = ut.variableInitializer(dt_save, 1.0)
    
    periodic = ut.variableInitializer(periodic, None)
    selfloop = ut.variableInitializer(selfloop, False)
    
    device = ut.variableInitializer(device, 'cuda' if torch.cuda.is_available() else 'cpu')
    
    save_directory_simulation = ut.variableInitializer(save_directory_simulation, '.')
    save_x_SDE = ut.variableInitializer(save_x_SDE, 'CTVicsek_SDE_traj.pt')
    save_t_SDE = ut.variableInitializer(save_t_SDE, 'CTVicsek_SDE_t_eval.pt')
    save_model = ut.variableInitializer(save_model, 'CTVicsek_SDE_model.pt')
    
    method_SDE = ut.variableInitializer(method_SDE, 'euler')
    noise_type = ut.variableInitializer(noise_type, 'general')
    sde_type = ut.variableInitializer(sde_type, 'ito')
    
    bm_levy = ut.variableInitializer(bm_levy, 'none')
    

    skipSimulate = ut.variableInitializer(skipSimulate, False)
    
    
    c_init = ut.variableInitializer(c_init, None)
    u0_init = ut.variableInitializer(u0_init, None)
    
    sigma_init = ut.variableInitializer(sigma_init, None)

    
    delayPredict = ut.variableInitializer(delayPredict, 1)
    dt_train = ut.variableInitializer(dt_train, dt_step)

    method_ODE = ut.variableInitializer(method_ODE, 'euler')
    N_epoch = ut.variableInitializer(N_epoch, 10)
    N_train_batch = ut.variableInitializer(N_train_batch, 8)
    N_batch_edgeUpdate = ut.variableInitializer(N_batch_edgeUpdate, 1)

    ratio_valid = ut.variableInitializer(ratio_valid, 1.0 / N_batch)
    ratio_test = ut.variableInitializer(ratio_test, 0.0)

    if split_seed_val is None:
        split_seed = torch.Generator()
    else:
        split_seed = torch.Generator().manual_seed(split_seed_val)
    
    lr = ut.variableInitializer(lr, 1e-3)
    lr_hyperSGD = ut.variableInitializer(lr_hyperSGD, 1e-3)
    vLoss_weight = ut.variableInitializer(vLoss_weight, 1.0)
    scoreLoss_weight = ut.variableInitializer(scoreLoss_weight, 1.0)
    useScore = ut.variableInitializer(useScore, False)
    
    save_directory_learning = ut.variableInitializer(save_directory_learning, '.')
    
    save_learned_model = ut.variableInitializer(save_learned_model, 'CTVicsek_parametric_learned_model.pt')
    save_loss_history = ut.variableInitializer(save_loss_history, 'CTVicsek_parametric_loss_history.pt')
    save_validloss_history = ut.variableInitializer(save_validloss_history, 'CTVicsek_parametric_validloss_history.pt')
    
    save_run_time_history = ut.variableInitializer(save_run_time_history, 'CTVicsek_parametric_run_time_history.npy')
    save_params = ut.variableInitializer(save_params, 'CTVicsek_parametric_parameters.npy')
    
    if not skipSimulate:
        os.makedirs(save_directory_simulation, exist_ok=True)
    os.makedirs(save_directory_learning, exist_ok=True)

    
    args_of_main = ut.getArgs()
    print(args_of_main)
    np.save(os.path.join(save_directory_learning, save_params), args_of_main)
    
    ut.dict2txt(os.path.join(save_directory_learning, os.path.splitext(save_params)[0]+'.txt'), args_of_main)
    
    if not skipSimulate:
        np.save(os.path.join(save_directory_simulation, save_params), args_of_main)
        ut.dict2txt(os.path.join(save_directory_simulation, os.path.splitext(save_params)[0]+'.txt'), args_of_main) 
    
    CTV_Module = ctv.interactionModule(u0, c, d, sigma, N_dim).to(device)
    edgeModule = sm.radiusgraphEdge(r0, periodic, selfloop, multiBatch=N_batch_edgeUpdate>1).to(device)
    
    CTV_SDEmodule = wm.dynamicGNDEmodule(CTV_Module, edgeModule, returnScore=False, 
                                        scorePostProcessModule=sm.pAndLogit2KLdiv(), scoreIntegrationModule=sm.scoreListModule(),
                                        N_multiBatch=N_batch_edgeUpdate).to(device)
    
    
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, N_dim]) * L, torch.rand([N_particles, N_dim-1]) * (2*np.pi)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'theta'], [N_dim, N_dim-1])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
        
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)

    
    
    print(t_save)
    
    CTV_SDEwrapper = wm.dynamicGSDEwrapper(CTV_SDEmodule, copy.deepcopy(graph_init).to(device), 
                                          ndataInOutModule=gu.multiVariableNdataInOut(['x', 'theta'], [N_dim, N_dim-1]), 
                                          derivativeInOutModule=gu.multiVariableNdataInOut(['v', 'w'], [N_dim, N_dim-1]),
                                          noise_type=noise_type, sde_type=sde_type).to(device)
    
    print(CTV_SDEwrapper.f(0, x0.to(device)))
    
    if not skipSimulate:
    
        bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                          size=(x0.shape[0], N_dim-1), dt=dt_step, levy_area_approximation=bm_levy, device=device)

        with torch.no_grad():
            y = sdeint(CTV_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)

        print(CTV_SDEwrapper.graph)

        y = y.to('cpu')
        if not(periodic is None):
            y[..., :N_dim] = torch.remainder(y[..., :N_dim], periodic)
        y[..., N_dim:] = torch.remainder(y[..., N_dim:], 2*np.pi)

        y = y.reshape((t_save.shape[0], N_batch, N_particles, 2*N_dim-1))

        torch.save(y, os.path.join(save_directory_simulation, save_x_SDE))

        torch.save(t_save.to('cpu'), os.path.join(save_directory_simulation, save_t_SDE))

        with open(os.path.join(save_directory_simulation, save_model), mode='wb') as f:
            cloudpickle.dump(CTV_SDEwrapper.to('cpu'), f)
    

    
    
    
    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.reset_parameter(u0_init, c_init, sigma_init)
    
    CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_returnScoreMode(useScore)
    
    print('Module before training : ', CTV_SDEwrapper.state_dict())
    
        
    optim = gdtuo.Adam(alpha=lr, beta1=0.9, beta2=0.999, log_eps=-8., optimizer=gdtuo.SGD(lr_hyperSGD))

    mw = gdtuo.ModuleWrapper(CTV_SDEwrapper, optimizer=optim)
    mw.initialize()
    
    
    
    neuralDE = NeuralODE(CTV_SDEwrapper, solver=method_ODE).to(device)
    
    
    
    t_pred_max = dt_save * float(delayPredict)
    
    t_learn_span = torch.arange(0, t_pred_max+dt_train, dt_train)
    t_learn_save = torch.tensor([t_pred_max])
    
    
    
    vicsek_dataset = ctv.myDataset(os.path.join(save_directory_simulation, save_x_SDE), N_dim=N_dim, delayTruth=delayPredict)
    vicsek_dataset.initialize()
    
    N_valid = int(vicsek_dataset.N_batch * ratio_valid)
    N_test = int(vicsek_dataset.N_batch * ratio_test)
    N_train = vicsek_dataset.N_batch - N_valid - N_test
    
    range_split = torch.utils.data.random_split(range(vicsek_dataset.N_batch), [N_train, N_valid, N_test], generator=split_seed)
    
    train_dataset = ctv.batchedSubset(vicsek_dataset, [i for i in range_split[0]])
    valid_dataset = ctv.batchedSubset(vicsek_dataset, [i for i in range_split[1]])
    test_dataset = ctv.batchedSubset(vicsek_dataset, [i for i in range_split[2]])
    
    train_loader = GraphDataLoader(train_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    if len(test_dataset) > 0:
        test_loader = GraphDataLoader(test_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    
    
    if periodic is None:
        lossFunc = ctv.myLoss(ut.euclidDistance_nonPeriodic(), N_dim=N_dim, useScore=useScore)
    else:
        lossFunc = ctv.myLoss(ut.euclidDistance_periodic(torch.tensor(periodic)), N_dim=N_dim, useScore=useScore)
        
    
    
    print('Number of snapshots in training data : ', train_dataset.__len__())
    
    
    
    best_valid_loss = np.inf
    
    print('epoch: trainLoss (xy, v, score), validLoss (xy, v, score), c, r_c, gamma, sigma, alpha, 1-beta1, 1-beta2, time[sec.]')
    
    loss_history = []
    valid_loss_history = []
    
    run_time_history = []
        
    start = time.time()
     
    for epoch in range(N_epoch):
        for graph, x_truth in train_loader:
            mw.begin()
            graph_batchsize = len(graph.batch_num_nodes())
            
            x_truth = x_truth.reshape([-1, x_truth.shape[-1]]).to(device)
            
            if useScore:
                CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                CTV_SDEwrapper.loadGraph(copy.deepcopy(graph).to(device))
                _ = CTV_SDEwrapper.f(1, x_truth)
                score_truth = torch.stack(CTV_SDEwrapper.score(), dim=1)
                CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(False)
            
            
            CTV_SDEwrapper.loadGraph(graph.to(device))
                        
            _, x_pred = neuralDE(CTV_SDEwrapper.ndataInOutModule.output(CTV_SDEwrapper.graph).to(device), 
                                 t_learn_span.to(device), save_at=t_learn_save.to(device))
            

            if useScore:
                if len(CTV_SDEwrapper.score())==0:
                    CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                    _ = CTV_SDEwrapper.f(t_learn_span.to(device)[-1], x_pred[0])
                    
                score_pred = torch.stack(CTV_SDEwrapper.score(), dim=1)
            
                xyloss, vloss, scoreloss = lossFunc(x_pred[0], x_truth, score_pred, score_truth)
                loss = xyloss + vLoss_weight * vloss + scoreLoss_weight * scoreloss
            else:
                xyloss, vloss = lossFunc(x_pred[0], x_truth)
                scoreloss = torch.full([1], torch.nan)
                loss = xyloss + vLoss_weight * vloss
                
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
                    CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                    CTV_SDEwrapper.loadGraph(copy.deepcopy(graph).to(device))
                    _ = CTV_SDEwrapper.f(1, x_truth)
                    score_truth = torch.stack(CTV_SDEwrapper.score(), dim=1)
                    CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(False)

                CTV_SDEwrapper.loadGraph(graph.to(device))
                _, x_pred = neuralDE(CTV_SDEwrapper.ndataInOutModule.output(CTV_SDEwrapper.graph).to(device), 
                                     t_learn_span.to(device), save_at=t_learn_save.to(device))
                
                if useScore:                
                    if len(CTV_SDEwrapper.score())==0:
                        CTV_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_forceUpdateMode(True)
                        _ = CTV_SDEwrapper.f(t_learn_span.to(device)[-1], x_pred[0])
                    score_pred = torch.stack(CTV_SDEwrapper.score(), dim=1)
                
                    valid_xyloss, valid_vloss, valid_scoreloss = lossFunc(x_pred[0], x_truth, score_pred, score_truth)
                    valid_xyloss_total = valid_xyloss_total + valid_xyloss * graph_batchsize
                    valid_vloss_total = valid_vloss_total + valid_vloss * graph_batchsize
                    valid_scoreloss_total = valid_scoreloss_total + valid_scoreloss * graph_batchsize
                    valid_loss = valid_loss + graph_batchsize * (valid_xyloss + vLoss_weight * valid_vloss + scoreLoss_weight * valid_scoreloss)
                    
                else:
                    valid_xyloss, valid_vloss = lossFunc(x_pred[0], x_truth)
                    valid_xyloss_total = valid_xyloss_total + valid_xyloss * graph_batchsize
                    valid_vloss_total = valid_vloss_total + valid_vloss * graph_batchsize
                    valid_scoreloss_total = torch.full([1], torch.nan)
                    valid_loss = valid_loss + graph_batchsize * (valid_xyloss + vLoss_weight * valid_vloss)
                    
                data_count = data_count + graph_batchsize
                
            valid_loss = valid_loss / data_count
            valid_xyloss_total = valid_xyloss_total / data_count
            valid_vloss_total = valid_vloss_total / data_count
            valid_scoreloss_total = valid_scoreloss_total / data_count
            valid_loss_history[-1] = [valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item()]
            
            run_time_history.append(time.time() - start)
            
            if valid_loss < best_valid_loss:
                CTV_SDEwrapper.deleteGraph()
                with open(os.path.join(save_directory_learning, save_learned_model), mode='wb') as f:
                    cloudpickle.dump(CTV_SDEwrapper.to('cpu'), f)
                best_valid_loss = valid_loss
                print('{}: {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.2e}, {:.2e}, {:.2e}, {:.3f} Best'.format(
                    epoch, loss.item(), xyloss.item(), vloss.item(), scoreloss.item(),
                    valid_loss.item(), valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.ctv.c().item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.ctv.r_c().item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.gamma.item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.sigma.item(),
                    mw.optimizer.parameters['alpha'].item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta1']).item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta2']).item(),
                    run_time_history[-1]))
            else:
                print('{}: {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f} ({:.3f}, {:.3f}, {:.2e}), {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.2e}, {:.2e}, {:.2e}, {:.3f}'.format(
                    epoch, loss.item(), xyloss.item(), vloss.item(), scoreloss.item(),
                    valid_loss.item(), valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.ctv.c().item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.ctv.r_c().item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.gamma.item(),
                    CTV_SDEwrapper.dynamicGNDEmodule.calc_module.sigma.item(),
                    mw.optimizer.parameters['alpha'].item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta1']).item(), 1-gdtuo.Adam.clamp(mw.optimizer.parameters['beta2']).item(),
                    run_time_history[-1]))
        
            torch.save(torch.tensor(loss_history), os.path.join(save_directory_learning, save_loss_history))

            torch.save(torch.tensor(valid_loss_history), os.path.join(save_directory_learning, save_validloss_history))
    
            np.save(os.path.join(save_directory_learning, save_run_time_history), run_time_history)

    
    
if __name__ == '__main__':

    parser = main_parser()
    
    args = parser.parse_args()
    
    parser2main(args)
    
        
