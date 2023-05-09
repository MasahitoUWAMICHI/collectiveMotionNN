import numpy as np

import torch
from torch import nn

import collections

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu


class springPotential(nn.Module):
    def __init__(self, c, r_c, p=2):
        super().__init__()
        
        self.logc = nn.Parameter(torch.tensor(np.log(c), requires_grad=True))
        self.logr_c = nn.Parameter(torch.tensor(np.log(r_c), requires_grad=True))

        self.p = p
        
    def c(self):
        return torch.exp(self.logc)
    
    def r_c(self):
        return torch.exp(self.logr_c)
        
    def potential(self, r):
        return self.c() * (r - self.r_c())**self.p

    def force(self, r):
        return -self.c() * self.p * (r - self.r_c())**(self.p - 1)

    
class interactionModule(nn.Module):
    def __init__(self, c, r_c, p=2, gamma=0.0, sigma=0.1, N_dim=2, periodic=None, positionName=None, velocityName=None, accelerationName=None, noiseName=None, messageName=None):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=True))
        
        self.sigma = nn.Parameter(torch.tensor(sigma, requires_grad=True))
        
        self.N_dim = N_dim
        
        self.prepare_sigma()
        
        self.sp = springPotential(c, r_c, p)
        
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
        
        
    def reset_parameter(self, c=None, r_c=None, gamma=None, sigma=None):
        if c is None:
            nn.init.uniform_(self.sp.logc)
        else:
            nn.init.constant_(self.sp.logc, np.log(c))
            
        if r_c is None:
            nn.init.uniform_(self.sp.logr_c)
        else:
            nn.init.constant_(self.sp.logr_c, np.log(r_c))
                        
        if gamma is None:
            nn.init.uniform_(self.gamma)
        else:
            nn.init.constant_(self.gamma, gamma)

        if sigma is None:
            nn.init.uniform_(self.sigma)
        else:
            nn.init.constant_(self.sigma, sigma)

        self.prepare_sigma()
        
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([self.N_dim,self.N_dim], device=self.sigma.device), self.sigma*torch.eye(self.N_dim, device=self.sigma.device)), dim=0)
            
    def def_nonPeriodic(self):
        self.distanceCalc = ut.euclidDistance_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = ut.euclidDistance_periodic(self.periodic)
        
    def def_dr(self):
        if self.periodic is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic()
            
    def calc_message(self, edges):
        dr = self.distanceCalc(edges.dst[self.positionName], edges.src[self.positionName])

        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {self.messageName: self.sp.force(abs_dr) * unit_dr}
    
    def aggregate_message(self, nodes):
        sum_force = torch.sum(nodes.mailbox[self.messageName], 1)
        return {self.accelerationName : sum_force}
        
    def f(self, t, g, args=None):
        g.update_all(self.calc_message, self.aggregate_message)
        g.ndata[self.accelerationName] = g.ndata[self.accelerationName] - self.gamma * g.ndata[self.velocityName]
        return g
      
    def g(self, t, g, args=None):
        self.prepare_sigma()
        g.ndata[self.noiseName] = self.sigmaMatrix.repeat(g.ndata[self.positionName].shape[0], 1, 1).to(g.device)
        return g
    
class interactionModule_nonParametric_acceleration(interactionModule):
    def __init__(self, gamma=None, sigma=None, N_dim=2, fNNshape=None, fBias=None, periodic=None, activationName=None, activationArgs=None, positionName=None, velocityName=None, accelerationName=None, noiseName=None, messageName=None):
        super().__init__(0.0, 0.0, 2, 0.0, 0.0, N_dim, periodic, positionName, velocityName, accelerationName, noiseName, messageName)
        self.reset_parameter(None, None, gamma, sigma)
        
        self.fNNshape = ut.variableInitializer(fNNshape, [128, 128, 128])
        
        self.fBias = ut.variableInitializer(fBias, True)
        
        self.init_f(activationName, activationArgs)
        
    def createActivation(self, act_name, args={}):
        args_str = ''
        key_exist = False
        for key in args.keys():
            key_exist = True
            args_str = args_str + key + '=args["' + key + '"],'

        if key_exist:
            args_str = args_str[:-1]

        return eval('nn.' + act_name + '(' + args_str + ')')
    
    def createNNsequence(self, N_in, NNshape, N_out, bias, activationName=None, activationArgs=None):
        activationName = ut.variableInitializer(activationName, 'ReLU')
        activationArgs = ut.variableInitializer(activationArgs, {})
        
        NNseq = collections.OrderedDict([])
        for i, NN_inout in enumerate(zip([N_in]+NNshape, NNshape+[N_out])):
            NNseq['Linear'+str(i)] = nn.Linear(NN_inout[0], NN_inout[1], bias=bias)
            NNseq[activationName+str(i)] = self.createActivation(activationName, activationArgs)
        NNseq.pop(activationName+str(i))
        
        return nn.Sequential(NNseq)
    
    def init_f(self, activationName=None, activationArgs=None):
        self.fNN = self.createNNsequence(1, self.fNNshape, 1, self.fBias, activationName, activationArgs)
        
    def make_reset_str(self, method, args, argsName, NNname='fNN'):
        initFunc_prefix = 'nn.init.{}(self.{}.'.format(method, NNname)
        initFunc_surfix = ''
        for key in args.keys():
            initFunc_surfix = initFunc_surfix + ','+key+'='+argsName+'["'+key+'"]'
        initFunc_surfix = initFunc_surfix + ')'
        return initFunc_prefix, initFunc_surfix        
        
    def reset_fNN(self, method_w=None, method_b=None, method_o=None, args_w={}, args_b={}, args_o={}, NNnames=['fNN'], zeroFinalLayer=False):
        for NNname in NNnames:
            if not method_w is None:
                initFunc_prefix_w, initFunc_surfix_w = self.make_reset_str(method_w, args_w, 'args_w', NNname)
            if not method_b is None:
                initFunc_prefix_b, initFunc_surfix_b = self.make_reset_str(method_b, args_b, 'args_b', NNname)
            if not method_o is None:
                initFunc_prefix_o, initFunc_surfix_o = self.make_reset_str(method_o, args_o, 'args_o', NNname)
            for key in eval('self.{}.state_dict().keys()'.format(NNname)):
                if key.endswith('weight'):
                    if not method_w is None:
                        eval(initFunc_prefix_w + key + initFunc_surfix_w)
                elif key.endswith('bias'):
                    if not method_b is None:
                        eval(initFunc_prefix_b + key + initFunc_surfix_b)
                else:
                    if not method_o is None:
                        eval(initFunc_prefix_o + key + initFunc_surfix_o)
            if zeroFinalLayer:
                print(NNname, 'zero initializing')
                initFunc_prefix_w, initFunc_surfix_w = self.make_reset_str('zeros_', {}, 'args_w', NNname+'[-1]')
                initFunc_prefix_b, initFunc_surfix_b = self.make_reset_str('zeros_', {}, 'args_b', NNname+'[-1]')
                initFunc_prefix_o, initFunc_surfix_o = self.make_reset_str('zeros_', {}, 'args_o', NNname+'[-1]')
                for key in eval('self.{}[-1].state_dict().keys()'.format(NNname)):
                    if key.endswith('weight'):
                        if not method_w is None:
                            eval(initFunc_prefix_w + key + initFunc_surfix_w)
                    elif key.endswith('bias'):
                        if not method_b is None:
                            eval(initFunc_prefix_b + key + initFunc_surfix_b)
                    else:
                        if not method_o is None:
                            eval(initFunc_prefix_o + key + initFunc_surfix_o)
            
        
    def calc_message(self, edges):
        dr = self.distanceCalc(edges.dst[self.positionName], edges.src[self.positionName])
        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)
        
        return {self.messageName: self.fNN(abs_dr) * unit_dr}
        
        
        
        
class interactionModule_nonParametric_2Dacceleration(interactionModule_nonParametric_acceleration):
    def __init__(self, gamma=None, sigma=None, N_dim=2, fNNshape=None, fBias=None, periodic=None, activationName=None, activationArgs=None, positionName=None, velocityName=None, accelerationName=None, noiseName=None, messageName=None):
        super().__init__(gamma, sigma, N_dim, fNNshape, fBias, periodic, activationName, activationArgs, positionName, velocityName, accelerationName, noiseName, messageName)
        
        self.init_f(activationName, activationArgs)
    
    def init_f(self, activationName=None, activationArgs=None):
        self.fNN = self.createNNsequence(self.N_dim, self.fNNshape, self.N_dim, self.fBias, activationName, activationArgs)
                
    def calc_message(self, edges):
        dr = self.distanceCalc(edges.dst[self.positionName], edges.src[self.positionName])
        
        return {self.messageName: self.fNN(dr)}
    
    
    
    
class interactionModule_nonParametric_2Dfull(interactionModule_nonParametric_2Dacceleration):
    def __init__(self, gamma=None, sigma=None, N_dim=2, fNNshape=None, fBias=None, f2NNshape=None, f2Bias=None, periodic=None, activationName=None, activationArgs=None, positionName=None, velocityName=None, accelerationName=None, noiseName=None, messageName=None):
        super().__init__(gamma, sigma, N_dim, fNNshape, fBias, periodic, activationName, activationArgs, positionName, velocityName, accelerationName, noiseName, messageName)
        
        self.f2NNshape = ut.variableInitializer(f2NNshape, [128, 128, 128])
        
        self.f2Bias = ut.variableInitializer(f2Bias, True)
        
        self.init_f2(activationName, activationArgs)
    
    def init_f2(self, activationName=None, activationArgs=None):
        self.f2NN = self.createNNsequence(self.N_dim, self.f2NNshape, self.N_dim, self.f2Bias, activationName, activationArgs)
    
    def f(self, t, g, args=None):
        g.update_all(self.calc_message, self.aggregate_message)
        g.ndata[self.accelerationName] = g.ndata[self.accelerationName] + self.f2NN(g.ndata[self.velocityName])
        return g
    
    
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, N_dim=2, len=None, delayTruth=1):
        super().__init__()
        
        self.dataPath = dataPath
        
        self.delayTruth = delayTruth
        
        self.N_dim = N_dim
                
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
        N_t, N_batch, N_particles, _ = xshape
        
        self.N_t = N_t
        self.N_batch = N_batch
        self.N_particles = N_particles
        
        self.t_max = self.N_t - self.extractDataLength + 1
        
        self.len = self.t_max * self.N_batch
    
    def calc_t_batch(self, index):
        return divmod(index, self.t_max)
    
    def calc_t_batch_subset(self, index, batchIndices_subset):
        batch_sub, t = divmod(index, self.t_max)
        return batchIndices_subset[batch_sub], t
    
    def from_t_batch(self, batch, t):
        _, x = self.loadData()
        
        gr = gu.make_disconnectedGraph(x[t, batch], gu.multiVariableNdataInOut(['x', 'v'], [self.N_dim, self.N_dim]))
        
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
    
    
    
class myLoss(nn.Module):
    def __init__(self, distanceCalc, N_dim=2, useScore=True):
        super().__init__()
        
        self.distanceCalc = distanceCalc
        self.useScore = useScore
                
        self.xyLoss = nn.MSELoss()
        self.vLoss = nn.MSELoss()
        
        self.N_dim = N_dim
        
        self.def_forward()
        
    def forward_score(self, x, y, score_x, score_y):
        dxy = self.distanceCalc(x[...,:self.N_dim], y[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        vLoss = self.xyLoss(x[...,self.N_dim:(2*self.N_dim)], y[...,self.N_dim:(2*self.N_dim)])
        scoreLoss = torch.mean(torch.square(torch.sum(score_x, dim=-1, keepdim=True) - score_y))
        return xyLoss, vLoss, scoreLoss
       
    def forward_noScore(self, x, y):
        dxy = self.distanceCalc(x[...,:self.N_dim], y[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        vLoss = self.xyLoss(x[...,self.N_dim:(2*self.N_dim)], y[...,self.N_dim:(2*self.N_dim)])
        return xyLoss, vLoss
       
    def def_forward(self):
        if self.useScore:
            self.forward = self.forward_score
        else:
            self.forward = self.forward_noScore
        
     
     
