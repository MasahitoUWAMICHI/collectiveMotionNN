import numpy as np
import torch

import copy

from torchsde import BrownianInterval, sdeint
from torchdyn.core import NeuralODE

import dgl

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.wrapper_modules as wm
import collectiveMotionNN.sample_modules as sm

import collectiveMotionNN.examples.springPotentialModel as spm

import argparse
from distutils.util import strtobool

import cloudpickle

import os
 

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=float)
    parser.add_argument('--v0', type=float)
    
    parser.add_argument('--N_dim', type=int)
    parser.add_argument('--N_particles', type=int)
    parser.add_argument('--N_batch', type=int)

    parser.add_argument('--t_max', type=float)
    parser.add_argument('--dt_step', type=float)
    parser.add_argument('--dt_save', type=float)
    
    parser.add_argument('--periodic', type=float)
    
    parser.add_argument('--device', type=str)
    
    parser.add_argument('--save_directory_simulation', type=str)
    parser.add_argument('--save_x', type=str)
    parser.add_argument('--save_t', type=str)
    parser.add_argument('--save_model', type=str)
    
    parser.add_argument('--isSDE', type=strtobool)    
    
    parser.add_argument('--sigma', type=float)    

    parser.add_argument('--method_SDE', type=str)
    parser.add_argument('--noise_type', type=str)
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--bm_levy', type=str)
        
    parser.add_argument('--method_ODE', type=str)
    
    parser.add_argument('--N_batch_edgeUpdate', type=int)
    
    parser.add_argument('--load_directory_learning', type=str)
    parser.add_argument('--load_learned_model', type=str)

    parser.add_argument('--save_params', type=str)
    
    return parser

def parser2main(args):
    main(sigma=args.sigma, L=args.L, v0=args.v0,
         N_dim=args.N_dim, N_particles=args.N_particles, N_batch=args.N_batch, 
         t_max=args.t_max, dt_step=args.dt_step, dt_save=args.dt_save, 
         periodic=args.periodic, 
         device=args.device,
         save_directory_simulation=args.save_directory_simulation,
         save_x=args.save_x, save_t=args.save_t, save_model=args.save_model,
         isSDE=args.isSDE,
         method_SDE=args.method_SDE, noise_type=args.noise_type, sde_type=args.sde_type, bm_levy=args.bm_levy,
         method_ODE=args.method_ODE, 
         N_batch_edgeUpdate=args.N_batch_edgeUpdate,
         load_directory_learning=args.load_directory_learning,
         load_learned_model=args.load_learned_model, 
         save_params=args.save_params)
    
def main(sigma=None, L=None, v0=None,
         N_dim=None, N_particles=None, N_batch=None, 
         t_max=None, dt_step=None, dt_save=None, 
         periodic=None, 
         device=None,
         save_directory_simulation=None,
         save_x=None, save_t=None, save_model=None,
         isSDE=None,
         method_SDE=None, noise_type=None, sde_type=None, bm_levy=None,
         method_ODE=None, 
         N_batch_edgeUpdate=None,
         load_directory_learning=None,
         load_learned_model=None, 
         save_params=None):

    sigma = ut.variableInitializer(sigma, 0.01)
    
    L = ut.variableInitializer(L, 5.0)
    v0 = ut.variableInitializer(v0, 0.01)
    
    N_dim = ut.variableInitializer(N_dim, int(2))
    N_particles = ut.variableInitializer(N_particles, int(100))
    N_batch = ut.variableInitializer(N_batch, int(5))
    
    t_max = ut.variableInitializer(t_max, 50.0)
    dt_step = ut.variableInitializer(dt_step, 0.1)
    dt_save = ut.variableInitializer(dt_save, 1.0)
    
    periodic = ut.variableInitializer(periodic, None)
    
    device = ut.variableInitializer(device, 'cuda' if torch.cuda.is_available() else 'cpu')
    
    save_directory_simulation = ut.variableInitializer(save_directory_simulation, '.')
    
    isSDE = ut.variableInitializer(isSDE, False)
    
    if isSDE:
        save_x = ut.variableInitializer(save_x, 'Spring_SDE_traj.pt')
        save_t = ut.variableInitializer(save_t, 'Spring_SDE_t_eval.pt')
        save_model = ut.variableInitializer(save_model, 'Spring_SDE_model.pt')
    else:
        save_x = ut.variableInitializer(save_x, 'Spring_ODE_traj.pt')
        save_t = ut.variableInitializer(save_t, 'Spring_ODE_t_eval.pt')
        save_model = ut.variableInitializer(save_model, 'Spring_ODE_model.pt')
        
    method_SDE = ut.variableInitializer(method_SDE, 'euler')
    noise_type = ut.variableInitializer(noise_type, 'general')
    sde_type = ut.variableInitializer(sde_type, 'ito')
    
    bm_levy = ut.variableInitializer(bm_levy, 'none')
    
    method_ODE = ut.variableInitializer(method_ODE, 'euler')
    
    N_batch_edgeUpdate = ut.variableInitializer(N_batch_edgeUpdate, 1)


    load_directory_learning = ut.variableInitializer(load_directory_learning, '.')
    
    load_learned_model = ut.variableInitializer(load_learned_model, 'Spring_nonParametric2Dfull_learned_model.pt')
    
    save_params = ut.variableInitializer(save_params, 'Spring_nonParametric2Dfull_reSimulate_parameters.npy')
    
    os.makedirs(save_directory_simulation, exist_ok=True)

    
    args_of_main = ut.getArgs()
    print(args_of_main)
    
    np.save(os.path.join(save_directory_simulation, save_params), args_of_main)
    ut.dict2txt(os.path.join(save_directory_simulation, os.path.splitext(save_params)[0]+'.txt'), args_of_main)
    
    
    with open(os.path.join(load_directory_learning, load_learned_model), mode='rb') as f:
        SP_SDEwrapper = cloudpickle.load(f)
    
    
    SP_SDEwrapper.dynamicGNDEmodule.calc_module.reset_parameter(sigma=sigma)
    
    SP_SDEwrapper.dynamicGNDEmodule.calc_module.periodic = periodic
    SP_SDEwrapper.dynamicGNDEmodule.calc_module.def_dr()
    
    SP_SDEwrapper.dynamicGNDEmodule.edgeRefresher.reset_N_multiBatch(N_batch_edgeUpdate)
    
    
    with open(os.path.join(save_directory_simulation, save_model), mode='wb') as f:
        cloudpickle.dump(SP_SDEwrapper.to('cpu'), f)
        
        
    x0 = []
    graph_init = []
    for i in range(N_batch):
        x0.append(torch.cat((torch.rand([N_particles, N_dim]) * L, (torch.rand([N_particles, N_dim])-0.5) * (2*v0)), dim=-1))
        graph_init.append(gu.make_disconnectedGraph(x0[i], gu.multiVariableNdataInOut(['x', 'v'], [N_dim, N_dim])))
    x0 = torch.concat(x0, dim=0)
    graph_init = dgl.batch(graph_init)
        
    t_span = torch.arange(0, t_max+dt_step, dt_step)
    t_save = torch.arange(0, t_max+dt_step, dt_save)

    SP_SDEwrapper.loadGraph(graph_init.to(device))
    

    
    if isSDE:
        bm = BrownianInterval(t0=t_save[0], t1=t_save[-1], 
                          size=(x0.shape[0], N_dim), dt=dt_step, levy_area_approximation=bm_levy, device=device)

        with torch.no_grad():
            y = sdeint(SP_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)

    else:
        neuralDE = NeuralODE(SP_SDEwrapper, solver=method_ODE).to(device)
        
        with torch.no_grad():
            _, y = neuralDE(SP_SDEwrapper.ndataInOutModule.output(SP_SDEwrapper.graph).to(device), 
                            t_span.to(device), save_at=t_save.to(device))
        
        
    print(SP_SDEwrapper.graph)

    y = y.to('cpu')
    if not(periodic is None):
        y[..., :N_dim] = torch.remainder(y[..., :N_dim], periodic)

    y = y.reshape((t_save.shape[0], N_batch, N_particles, 2*N_dim))

    torch.save(y, os.path.join(save_directory_simulation, save_x))

    torch.save(t_save.to('cpu'), os.path.join(save_directory_simulation, save_t))

    SP_SDEwrapper.deleteGraph()


    
if __name__ == '__main__':

    parser = main_parser()
    
    args = parser.parse_args()
    
    parser2main(args)
    
        

