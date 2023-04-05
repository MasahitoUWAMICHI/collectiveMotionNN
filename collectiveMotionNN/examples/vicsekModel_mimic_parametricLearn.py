import numpy as np
import torch
from torch import nn

import copy

from torchsde import BrownianInterval, sdeint

import dgl
import dgl.function as fn

from dgl.dataloading import GraphDataLoader

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu
import collectiveMotionNN.module as mo

import argparse
from distutils.util import strtobool

import cloudpickle

class interactionModule(nn.Module):
    def __init__(self, u0, w0, sigma=0.1, d=1, positionName=None, velocityName=None, polarityName=None, torqueName=None, noiseName=None, messageName=None):
        super().__init__()
        
        self.v0 = nn.Parameter(torch.tensor(v0, requires_grad=True))
        self.w0 = nn.Parameter(torch.tensor(w0, requires_grad=True))

        self.sigma = nn.Parameter(torch.tensor(sigma, requires_grad=True))
        
        self.d = d
        
        self.prepare_sigma()
                    
        self.positionName = ut.variableInitializer(positionName, 'x')
        self.velocityName = ut.variableInitializer(velocityName, 'v')
        self.polarityName = ut.variableInitializer(polarityName, 'theta')
        self.torqueName = ut.variableInitializer(torqueName, 'w')
        self.noiseName = ut.variableInitializer(noiseName, 'sigma')
        
        self.messageName = ut.variableInitializer(messageName, 'm')
        
    def reset_parameter(self):
        nn.init.uniform_(self.v0)
        nn.init.uniform_(self.w0)
        nn.init.uniform_(self.sigma)
        
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([2,1]), self.sigma*torch.ones([1,1])), dim=0)
            
    def calc_message(self, edges):
        dtheta = (edges.src[self.polarityName] - edges.dst[self.polarityName]) * self.d
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
    
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, len=None, delayTruth=1):
        super().__init__()
        
        self.dataPath = dataPath
        
        self.delayTruth = delayTruth
                
        if len is None:
            self.initialize()
        else:
            self.len = len
        
    def __len__(self):
        return self.len
      
    def loadData(self):
        x = torch.load(self.dataPath)
        return x.shape, x
      
    def initialize(self):
        self.extractDataLength = 1 + self.delayTruth
        
        xshape, _ = self.loadData()
        N_t, N_batch, N_particles, N_dim = xshape
        
        self.N_t = N_t.item()
        self.N_batch = N_batch.item()
        self.N_particles = N_particles.item()
        self.N_dim = N_dim.item()
        
        self.t_max = self.N_t - self.extractDataLength + 1
        
        self.len = self.t_max * self.N_batch
    
    def calc_t_batch(self, index):
        return divmod(index, self.t_max)
    
    def calc_t_batch_subset(self, index, batchIndices_subset):
        batch_sub, t = divmod(index, self.t_max)
        return batchIndices_subset[batch_sub], t
    
    def from_t_batch(self, batch, t):
        _, x = self.loadData()
        
        gr = gu.make_disconnectedGraph(x[t, batch], gu.multiVariableNdataInOut(['x', 'theta'], [2, 1]))
        
        x_truth = x[t+self.delayTruth, batch]
        
        return gr, x_truth
    
    def __getitem__(self, index):
        batch, t = self.calc_t_batch(index)
        return self.from_t_batch(batch, t)
    
class batchedSubset(torch.utils.data.Subset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices of batch in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        batch, t = self.dataset.calc_t_batch_subset(idx, self.indices)
        return self.dataset.from_t_batch(batch, t)

    def __len__(self):
        return len(self.indices) * self.dataset.t_max
    
    
    
class cosLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 1 - torch.cos(x - y).mean()
    
class myLoss(nn.Module):
    def __init__(self, distanceCalc):
        super().__init__()
        
        self.distanceCalc = distanceCalc
                
        self.xyLoss = nn.MSELoss()
        self.thetaLoss = cosLoss()
        
    def forward(self, x, y):
        dxy = self.distanceCalc(x[..., :2], y[..., :2])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x[..., 2], y[..., 2])
        return xyLoss, thetaLoss
    
    
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--v0', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--d', type=float)
    
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
    
    
    
    parser.add_argument('--delayPredict', type=int)
    parser.add_argument('--dt_train', type=float)
    
    parser.add_argument('--method_ODE', type=str)
    parser.add_argument('--N_epoch', type=int)
    parser.add_argument('--N_train_batch', type=int)
    
    parser.add_argument('--ratio_valid', type=float)
    parser.add_argument('--ratio_test', type=float)
    parser.add_argument('--split_seed', type=int)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--thetaLoss_weight', type=float)
    
    parser.add_argument('--save_learned_model', type=str)
    parser.add_argument('--save_loss_history', type=str)
    parser.add_argument('--save_validloss_history', type=str)
    
    
    args = parser.parse_args()
    
    
    v0 = ut.variableInitializer(args.v0, 0.03)
    w0 = ut.variableInitializer(args.w0, 1.0)
    
    sigma = ut.variableInitializer(args.sigma, 0.3)
        
    d = ut.variableInitializer(args.d, 1)
    
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
    
    
    
    delayPredict = ut.variableInitializer(args.delayPredict, 1)
    dt_train = ut.variableInitializer(args.dt_train, dt_step)

    method_ODE = ut.variableInitializer(args.method_ODE, 'euler')
    N_epoch = ut.variableInitializer(args.N_epoch, 10)
    N_train_batch = ut.variableInitializer(args.N_train_batch, 8)

    ratio_valid = ut.variableInitializer(args.ratio_valid, 0.0)
    ratio_test = ut.variableInitializer(args.ratio_test, 0.0)
    ratio_train = 1.0 - ratio_valid - ratio_test

    if args.split_seed is None:
        split_seed = torch.Generator()
    else:
        split_seed = torch.Generator().manual_seed(args.split_seed)
    
    lr = ut.variableInitializer(args.lr, 1e-3)
    thetaLoss_weight = ut.variableInitializer(args.thetaLoss_weight, 1.0)
    
    save_learned_model = ut.variableInitializer(args.save_learned_model, 'Vicsek_parametric_learned_model.pt')
    save_loss_history = ut.variableInitializer(args.save_loss_history, 'Vicsek_parametric_loss_history.pt')
    save_validloss_history = ut.variableInitializer(args.save_validloss_history, 'Vicsek_parametric_validloss_history.pt')
    
    
    
    
    
    Vicsek_Module = interactionModule(v0, w0, sigma, d).to(device)
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
                      size=(x0.shape[0], 1), dt=dt_step, device=device)
  
    with torch.no_grad():
        y = sdeint(Vicsek_SDEwrapper, x0.to(device), t_save, bm=bm, dt=dt_step, method=method_SDE)
    
    print(Vicsek_SDEwrapper.graph)
    
    y = y.to('cpu')
    if not(periodic is None):
        y[..., :2] = torch.remainder(y[..., :2], periodic)
    
    y = y.reshape((t_save.shape[0], N_batch, N_particles, 3))

    torch.save(y, save_x_SDE)

    torch.save(t_save.to('cpu'), save_t_SDE)
    
    with open(save_model, mode='wb') as f:
        cloudpickle.dump(Vicsek_SDEwrapper.to('cpu'), f)
    

    
    
    
    Vicsek_SDEwrapper.dynamicGNDEmodule.calc_module.reset_parameter()
    
    neuralDE = NeuralODE(Vicsek_SDEwrapper, solver=method_ODE).to(device)
    
    t_pred_max = dt_save * float(delayPredict)
    
    t_learn_span = torch.arange(0, t_pred_max+dt_train, dt_train)
    t_learn_save = torch.tensor([t_pred_max])
    
    
    
    vicsek_dataset = myDataset(save_x_SDE, delayTruth=delayPredict)
    vicsek_dataset.initialize()
    
    range_split = random_split(range(vicsek_dataset.N_batch), [ratio_train, ratio_valid, ratio_test], generator=generator)
    
    train_dataset = batchedSubset(vicsek_dataset, [i for i in range_split[0]])
    valid_dataset = batchedSubset(vicsek_dataset, [i for i in range_split[1]])
    test_dataset = batchedSubset(vicsek_dataset, [i for i in range_split[2]])
    
    train_loader = GraphDataLoader(train_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    test_loader = GraphDataLoader(test_dataset, batch_size=N_train_batch, drop_last=False, shuffle=True, pin_memory=True)
    
    
    if periodic is None:
        lossFunc = myLoss(ut.euclidDistance_nonPeriodic())
    else:
        lossFunc = myLoss(ut.euclidDistance_periodic(torch.tensor(periodic)))
        
    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    
    best_valid_loss = np.inf
    
    print('epoch: trainLoss (xy, theta), validLoss (xy, theta)')
    
    loss_history = []
    valid_loss_history = []
    
    saved_model = Vicsek_SDEwrapper.cpu().detach()
    
    for epoch in range(N_epoch):
        for graph, x_truth in train_loader:
            _, x_pred = neuralDE(graph.to(device), t_learn_span.to(device), save_at=t_learn_save.to(device))
            xyloss, thetaloss = lossFunc(x_pred[0], x_truth)
            loss = xyloss + thetaLoss_weight * thetaloss
            loss_history.append([xyloss.item(), thetaloss.item()])
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            valid_loss = 0
            valid_xyloss_total = 0
            valid_thetaloss_total = 0
            data_count = 0
            
            for graph, x_truth in valid_loader:
                _, x_pred = neuralDE(graph.to(device), t_learn_span.to(device), save_at=t_learn_save.to(device))
                valid_xyloss, valid_thetaloss = lossFunc(x_pred[0], x_truth)
                valid_xyloss_total = valid_xyloss_total + x_truth.shape[0] * valid_xyloss
                valid_thetaloss_total = valid_thetaloss_total + x_truth.shape[0] * valid_thetaloss
                valid_loss = valid_loss + x_truth.shape[0] * (valid_xyloss + thetaLoss_weight * valid_thetaloss)
                data_count = data_count + x_truth.shape[0]
                
            valid_loss = valid_loss / data_count
            valid_xyloss_total = valid_xyloss_total / data_count
            valid_thetaloss_total = valid_thetaloss_total / data_count
            valid_loss_history.append([valid_xyloss.item(), valid_thetaloss.item()])
            
        if valid_loss < best_loss:
            saved_model = Vicsek_SDEwrapper.cpu().detach()
            best_loss = valid_loss
            print('{}: {:.3f} ({:.3f}, {:.3f}), {:.3f} ({:.3f}, {:.3f}) Best'.format(
                epoch, loss.item(), xyloss.item(), thetaloss.item(), valid_loss.item(), valid_xyloss_total.item(), valid_thetaloss_total.item()))
        else:
            print('{}: {:.3f} ({:.3f}, {:.3f}), {:.3f} ({:.3f}, {:.3f})'.format(
                epoch, loss.item(), xyloss.item(), thetaloss.item(), valid_loss.item(), valid_xyloss_total.item(), valid_thetaloss_total.item()))
        
    torch.save(torch.tensor(loss_history), save_loss_history)

    torch.save(torch.tensor(valid_loss_history), save_validloss_history)
    
    with open(save_learned_model, mode='wb') as f:
        cloudpickle.dump(saved_model.to('cpu'), f)
        

