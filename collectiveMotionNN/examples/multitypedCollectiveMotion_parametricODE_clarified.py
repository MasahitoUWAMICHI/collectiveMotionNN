import numpy as np
import torch

from torchdyn.core import NeuralODE

import dgl.function as fn

import torch_optimizer as t_opt

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.wrapper_modules as wm
import collectiveMotionNN.sample_modules as sm

import collectiveMotionNN.examples.multitypedCollectiveMotionFunctions as mcmf
import collectiveMotionNN.examples.multitypedCollectiveMotion_utils as mcm_ut

import argparse
from distutils.util import strtobool

import cloudpickle

import time
import os
 

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
    parser.add_argument('--r_c_init', type=float)
    parser.add_argument('--gamma_init', type=float)
    parser.add_argument('--sigma_init', type=float)
    
    parser.add_argument('--bm_levy', type=str)
    
    parser.add_argument('--delayPredict', type=int)
    parser.add_argument('--dt_train', type=float)
    
    parser.add_argument('--method_ODE', type=str)
    parser.add_argument('--N_epoch', type=int)
    parser.add_argument('--N_train_batch', type=int)
    parser.add_argument('--N_batch_edgeUpdate', type=int)
    parser.add_argument('--N_train_minibatch_integrated', type=int)
    
    parser.add_argument('--ratio_valid', type=float)
    parser.add_argument('--ratio_test', type=float)
    parser.add_argument('--split_seed_val', type=int)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--optimName', type=str)
    parser.add_argument('--optimArgs', type=dict)
    parser.add_argument('--highOrderGrad', type=strtobool)

    parser.add_argument('--lrSchedulerName', type=str)
    parser.add_argument('--lrSchedulerArgs', type=dict)
    
    parser.add_argument('--vLoss_weight', type=float)
    parser.add_argument('--scoreLoss_weight', type=float)
    parser.add_argument('--useScore', type=strtobool)
    parser.add_argument('--nondimensionalLoss', type=strtobool)
    
    parser.add_argument('--save_directory_learning', type=str)
    parser.add_argument('--save_learned_model', type=str)
    parser.add_argument('--save_loss_history', type=str)
    parser.add_argument('--save_validloss_history', type=str)
    parser.add_argument('--save_lr_history', type=str)    
    parser.add_argument('--save_run_time_history', type=str)
    parser.add_argument('--save_params', type=str)
    
    return parser

def parser2main(args):
    main(c=args.c, r_c=args.r_c, p=args.p, gamma=args.gamma, sigma=args.sigma, r0=args.r0, L=args.L, v0=args.v0,
         N_dim=args.N_dim, N_particles=args.N_particles, N_batch=args.N_batch, 
         t_max=args.t_max, dt_step=args.dt_step, dt_save=args.dt_save, 
         periodic=args.periodic, selfloop=args.selfloop, 
         device=args.device,
         save_directory_simulation=args.save_directory_simulation,
         save_x_SDE=args.save_x_SDE, save_t_SDE=args.save_t_SDE, save_model=args.save_model,
         method_SDE=args.method_SDE, noise_type=args.noise_type, sde_type=args.sde_type, bm_levy=args.bm_levy,
         skipSimulate=args.skipSimulate,
         c_init=args.c_init, r_c_init=args.r_c_init, 
         gamma_init=args.gamma_init, sigma_init=args.sigma_init, 
         delayPredict=args.delayPredict, dt_train=args.dt_train, 
         method_ODE=args.method_ODE, 
         N_epoch=args.N_epoch, N_train_batch=args.N_train_batch, N_batch_edgeUpdate=args.N_batch_edgeUpdate,
         N_train_minibatch_integrated=args.N_train_minibatch_integrated, 
         ratio_valid=args.ratio_valid, ratio_test=args.ratio_test,
         split_seed_val=args.split_seed_val,
         lr=args.lr, optimName=args.optimName, optimArgs=args.optimArgs, highOrderGrad=args.highOrderGrad,
         lrSchedulerName=args.lrSchedulerName, lrSchedulerArgs=args.lrSchedulerArgs,
         vLoss_weight=args.vLoss_weight, scoreLoss_weight=args.scoreLoss_weight, 
         useScore=args.useScore,
         nondimensionalLoss=args.nondimensionalLoss,
         save_directory_learning=args.save_directory_learning,
         save_learned_model=args.save_learned_model, 
         save_loss_history=args.save_loss_history, save_validloss_history=args.save_validloss_history,
         save_lr_history=args.save_lr_history,
         save_run_time_history=args.save_run_time_history,
         save_params=args.save_params)
    
def main(c=None, r_c=None, p=None, gamma=None, sigma=None, r0=None, L=None, v0=None,
         N_dim=None, N_particles=None, N_batch=None, 
         t_max=None, dt_step=None, dt_save=None, 
         periodic=None, selfloop=None, 
         device=None,
         save_directory_simulation=None,
         save_x_SDE=None, save_t_SDE=None, save_model=None,
         method_SDE=None, noise_type=None, sde_type=None, bm_levy=None,
         skipSimulate=None,
         c_init=None, r_c_init=None,
         gamma_init=None, sigma_init=None,
         delayPredict=None, dt_train=None, 
         method_ODE=None, 
         N_epoch=None, N_train_batch=None, N_batch_edgeUpdate=None,
         N_train_minibatch_integrated=None,
         ratio_valid=None, ratio_test=None,
         split_seed_val=None,
         lr=None, optimName=None, optimArgs=None, highOrderGrad=None,
         lrSchedulerName=None, lrSchedulerArgs=None,
         vLoss_weight=None, scoreLoss_weight=None, 
         useScore=None,
         nondimensionalLoss=None,
         save_directory_learning=None,
         save_learned_model=None, 
         save_loss_history=None, save_validloss_history=None,
         save_lr_history=None,
         save_run_time_history=None,
         save_params=None):

    c = ut.variableInitializer(c, 0.01)
    r_c = ut.variableInitializer(r_c, 1.0)
    p = ut.variableInitializer(p, 2.0)
    
    gamma = ut.variableInitializer(gamma, 0.1)
    sigma = ut.variableInitializer(sigma, 0.01)
    
    
    r0 = ut.variableInitializer(r0, 5.0)
    L = ut.variableInitializer(L, 5.0)
    v0 = ut.variableInitializer(v0, 0.01)
    
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
    save_x_SDE = ut.variableInitializer(save_x_SDE, 'Mcm_SDE_traj.pt')
    save_t_SDE = ut.variableInitializer(save_t_SDE, 'Mcm_SDE_t_eval.pt')
    save_model = ut.variableInitializer(save_model, 'Mcm_SDE_model.pt')
    
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
    N_batch_edgeUpdate = ut.variableInitializer(N_batch_edgeUpdate, 1)
    N_train_minibatch_integrated = ut.variableInitializer(N_train_minibatch_integrated, 1)

    ratio_valid = ut.variableInitializer(ratio_valid, 1.0 / N_batch)
    ratio_test = ut.variableInitializer(ratio_test, 0.0)

    if split_seed_val is None:
        split_seed = torch.Generator()
    else:
        split_seed = torch.Generator().manual_seed(split_seed_val)
    
    lr = ut.variableInitializer(lr, 1e-3)
    optimName = ut.variableInitializer(optimName, 'Lamb')
    optimArgs = ut.variableInitializer(optimArgs, {})
    highOrderGrad = ut.variableInitializer(highOrderGrad, False)
    
    lrSchedulerName = ut.variableInitializer(lrSchedulerName, None)
    lrSchedulerArgs =  ut.variableInitializer(lrSchedulerArgs, {})
    
    vLoss_weight = ut.variableInitializer(vLoss_weight, 1.0)
    scoreLoss_weight = ut.variableInitializer(scoreLoss_weight, 1.0)
    useScore = ut.variableInitializer(useScore, False)

    nondimensionalLoss = ut.variableInitializer(nondimensionalLoss, False)
    
    save_directory_learning = ut.variableInitializer(save_directory_learning, '.')
    
    save_learned_model = ut.variableInitializer(save_learned_model, 'Mcm_parametric_learned_model.pt')
    save_loss_history = ut.variableInitializer(save_loss_history, 'Mcm_parametric_loss_history.pt')
    save_validloss_history = ut.variableInitializer(save_validloss_history, 'Mcm_parametric_validloss_history.pt')
    save_lr_history = ut.variableInitializer(save_lr_history, 'Mcm_parametric_lr_history.pt')
    
    save_run_time_history = ut.variableInitializer(save_run_time_history, 'Mcm_parametric_run_time_history.npy')
    save_params = ut.variableInitializer(save_params, 'Mcm_parametric_parameters.npy')
        
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

    save_history = os.path.join(save_directory_learning, os.path.splitext(save_loss_history)[0]+'.txt')
    
    SP_Module = mcmf.interactionModule(c, r_c, p, gamma, sigma, N_dim, periodic).to(device)
    edgeModule = sm.radiusgraphEdge(r0, periodic, selfloop, multiBatch=N_batch_edgeUpdate>1).to(device)
    
    
    x0, graph_init = mcm_ut.init_graph(L, v0, N_particles, N_dim, N_batch)
        

    SP_SDEmodule, SP_SDEwrapper = mcm_ut.init_SDEwrappers(SP_Module, edgeModule, graph_init, device, noise_type, sde_type, N_batch_edgeUpdate=1, 
                                                          scorePostProcessModule=sm.pAndLogit2KLdiv(), 
                                                          scoreIntegrationModule=sm.scoreListModule())
    
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)
    
    if not skipSimulate:
    
        y = mcm_ut.run_SDEsimulate(SP_SDEwrapper, x0, t_save, dt_step, device, method_SDE, bm_levy)
        
        y = y.reshape((t_save.shape[0], N_batch, N_particles, 2*N_dim))

        torch.save(y, os.path.join(save_directory_simulation, save_x_SDE))

        torch.save(t_save.to('cpu'), os.path.join(save_directory_simulation, save_t_SDE))
        
        SP_SDEwrapper.deleteGraph()

        with open(os.path.join(save_directory_simulation, save_model), mode='wb') as f:
            cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
    

    
    SP_SDEwrapper.dynamicGNDEmodule.calc_module.reset_parameter(c_init, r_c_init, gamma_init, sigma_init)
    
    SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_returnScoreMode(useScore)
    
    
    
    print('Module before training : ', SP_SDEwrapper.state_dict())
    
    optim_str = 't_opt.' + optimName + '(SP_SDEwrapper.parameters(),lr={},'.format(lr)
    for key in optimArgs.keys():
        optim_str = optim_str + key + '=optimArgs["' + key + '"],'
    optim_str = optim_str[:-1] + ')'
    optimizer = eval(optim_str)
    
    flg_scheduled = not(lrSchedulerName is None)
    
    if flg_scheduled:
        schedulerStr = 'torch.optim.lr_scheduler.' + lrSchedulerName + '(optimizer'
        for key in lrSchedulerArgs.keys():
            schedulerStr = schedulerStr + ',' + key + '=lrSchedulerArgs["' + key + '"]'
        schedulerStr = schedulerStr + ')'
        scheduler = eval(schedulerStr)
    
    
    neuralDE = NeuralODE(SP_SDEwrapper, solver=method_ODE).to(device)
    
    
    
    t_pred_max = dt_save * float(delayPredict)
    
    t_learn_span = torch.arange(0, t_pred_max+dt_train, dt_train)
    t_learn_save = torch.tensor([t_pred_max])
    
    

    train_loader, valid_loader, test_loader = mcm_ut.makeGraphDataLoader(os.path.join(save_directory_simulation, save_x_SDE),
                                                                         N_dim, delayPredict, ratio_valid, ratio_test, 
                                                                         split_seed=split_seed, batch_size=N_train_batch, 
                                                                         drop_last=False, shuffle=True, pin_memory=True)
    
    print('Number of snapshots in training data : ', train_loader.dataset.__len__())


    lossFunc = mcm_ut.makeLossFunc(N_dim, useScore, periodic, nondimensionalLoss)

    best_valid_loss = np.inf
    
    text_for_print = 'epoch: trainLoss (xy, v, score), validLoss (xy, v, score), c, r_c, gamma, sigma, time[sec.]'
    with open(save_history, 'w') as f:
        f.write(text_for_print)
    print(text_for_print)

    
    loss_history = []
    valid_loss_history = []
    
    run_time_history = []
    lr_history = []
        
    start = time.time()
     
    for epoch in range(N_epoch):
        i_minibatch = 0
        flg_zerograd = True
        SP_SDEwrapper.train()
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_history.append(lrs)
        for i_minibatch, gx in enumerate(train_loader, 1):
            graph, x_truth = gx
            if flg_zerograd:
                optimizer.zero_grad()
            
            x_pred, x_truth = mcm_ut.run_ODEsimulate(neuralDE, SP_SDEwrapper, graph, x_truth, device, t_learn_span, t_learn_save, useScore)

            loss, xyloss, vloss, scoreloss = mcm_ut.calcLoss(lossFunc, x_pred, x_truth, vLoss_weight, device, useScore, SP_SDEwrapper, scoreLoss_weight, t_learn_span)
                
            loss_history.append([xyloss.item(), vloss.item(), scoreloss.item()])
            valid_loss_history.append([np.nan, np.nan, np.nan])
            loss.backward(create_graph=highOrderGrad)
            if i_minibatch % N_train_minibatch_integrated == 0:
                optimizer.step()
                flg_zerograd = True
            else:
                flg_zerograd = False
        if i_minibatch % N_train_minibatch_integrated > 0:
            optimizer.step()
        if flg_scheduled:
            scheduler.step()
        
        SP_SDEwrapper.eval()
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            valid_loss = 0
            valid_xyloss_total = 0
            valid_vloss_total = 0
            valid_scoreloss_total = 0
            data_count = 0
            
            for graph, x_truth in valid_loader:
                graph_batchsize = len(graph.batch_num_nodes())
                
                x_pred, x_truth = mcm_ut.run_ODEsimulate(neuralDE, SP_SDEwrapper, graph, x_truth, device, t_learn_span, t_learn_save, useScore)

                valid_loss_batch, valid_xyloss, valid_vloss, valid_scoreloss = mcm_ut.calcLoss(lossFunc, x_pred, x_truth, vLoss_weight, device, useScore, SP_SDEwrapper, scoreLoss_weight, t_learn_span)
                    
                valid_xyloss_total = valid_xyloss_total + valid_xyloss * graph_batchsize
                valid_vloss_total = valid_vloss_total + valid_vloss * graph_batchsize
                valid_scoreloss_total = valid_scoreloss_total + valid_scoreloss * graph_batchsize
                valid_loss = valid_loss + valid_loss_batch * graph_batchsize
                data_count = data_count + graph_batchsize
                
            valid_loss = valid_loss / data_count
            valid_xyloss_total = valid_xyloss_total / data_count
            valid_vloss_total = valid_vloss_total / data_count
            valid_scoreloss_total = valid_scoreloss_total / data_count
            valid_loss_history[-1] = [valid_xyloss_total.item(), valid_vloss_total.item(), valid_scoreloss_total.item()]
            
            run_time_history.append(time.time() - start)

            info_txt = '{}: '.format(epoch)
            info_txt = info_txt + '{:.3f} ({:.3f}, {:.3f}, {:.2e}), '.format(loss.item(), xyloss.item(), 
                                                                             vloss.item(), scoreloss.item())
            info_txt = info_txt + '{:.3f} ({:.3f}, {:.3f}, {:.2e}), '.format(valid_loss.item(), valid_xyloss_total.item(), 
                                                                             valid_vloss_total.item(), valid_scoreloss_total.item())
            info_txt = info_txt + '{:.3f}, {:.3f}, {:.3f}, {:.3f}, '.format(SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.c().item(),
                                                                            SP_SDEwrapper.dynamicGNDEmodule.calc_module.sp.r_c().item(),
                                                                            SP_SDEwrapper.dynamicGNDEmodule.calc_module.gamma.item(),
                                                                            SP_SDEwrapper.dynamicGNDEmodule.calc_module.sigma.item())
            info_txt = info_txt + '{:.3f}'.format(run_time_history[-1])
            
            if valid_loss < best_valid_loss:
                SP_SDEwrapper.deleteGraph()
                with open(os.path.join(save_directory_learning, save_learned_model), mode='wb') as f:
                    cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
                SP_SDEwrapper.to(device)
                best_valid_loss = valid_loss
                info_txt = info_txt + ' Best'

            print(info_txt)
            with open(save_history, 'w') as f:
                f.write(info_txt)
     
            torch.cuda.empty_cache()
        
            torch.save(torch.tensor(loss_history), os.path.join(save_directory_learning, save_loss_history))

            torch.save(torch.tensor(valid_loss_history), os.path.join(save_directory_learning, save_validloss_history))
    
            np.save(os.path.join(save_directory_learning, save_run_time_history), run_time_history)

            np.save(os.path.join(save_directory_learning, save_lr_history), lr_history)
    
    
if __name__ == '__main__':

    parser = main_parser()
    
    args = parser.parse_args()
    
    parser2main(args)
    
        
