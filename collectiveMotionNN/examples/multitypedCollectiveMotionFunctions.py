import numpy as np
import torch
from torch import nn

from scipy import special
from torch.autograd import Function

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu

import collections



## prepare functions

class torch_knFunction(Function):
    @staticmethod
    def forward(ctx, input, n):
        ctx.save_for_backward(input)
        ctx.n = n
        numpy_input = input.cpu().detach().numpy()
        result = special.kn(n, numpy_input)
        return torch.from_numpy(result).to(input.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.cpu().detach().numpy()
        input, = ctx.saved_tensors
        n = ctx.n
        numpy_input = input.cpu().detach().numpy()
        if n==0:
            grad_kn = -special.kn(1, numpy_input)
        else:
            grad_kn = -(special.kn(n-1, numpy_input) + special.kn(n+1, numpy_input))/2
        result = numpy_go * grad_kn
        return torch.from_numpy(result).to(grad_output.device), None

class torch_kn(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def forward(self, input):
        return torch_knFunction.apply(input, self.n)

class torch_kn_cutoff(nn.Module):
    def __init__(self, n, cutoff):
        super().__init__()
        self.n = n
        self.cutoff = cutoff
        
        self.cutoff_val = torch_knFunction.apply(self.cutoff, self.n)
        
        self.cutoff_module = nn.ReLU()
        
    def forward(self, input):
        return self.cutoff_module(torch_knFunction.apply(input, self.n) - self.cutoff_val)

class J_chemoattractant2D(nn.Module):
    def __init__(self, kappa, cutoff):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(kappa, requires_grad=True))
        
        self.cutoff = torch.tensor(cutoff)
        self.set_k1()

    def set_k1(self):
        if self.cutoff > 0:
            self.k1 = torch_kn_cutoff(1, self.cutoff)
        else:
            self.k1 = torch_kn(1)
    
    def forward(self, input):
        return self.k1(self.kappa * input) * (self.kappa/(2*np.pi))
    
class J_contactFollowing(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, xy, p):
        return (1 + torch.sum(xy * p, dim=-1, keepdim=True)) / 2

class J_contactInhibitionOfLocomotion(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(r, requires_grad=True))
    
    def forward(self, d):
        return (self.r/d) - 1


## make module

class interactionModule(nn.Module):
    def __init__(self, params, sigma=0.1, N_dim=2, N_celltypes=2, positionName=None, velocityName=None, polarityName=None, torqueName=None, noiseName=None, celltypeName=None, torquemessageName=None, velocitymessageName=None, periodic=None):
        super().__init__()
        
        self.sigma = nn.Parameter(torch.tensor(sigma, requires_grad=True))
        
        self.N_dim = N_dim

        self.N_celltypes = N_celltypes
        
        self.prepare_sigma()

        
        self.J_chem = J_chemoattractant2D(params['kappa'], params['cutoff'])
        self.J_CF = J_contactFollowing()
        self.J_CIL = J_contactInhibitionOfLocomotion(params['r'])

        self.u0 = nn.Parameter(torch.tensor(params['u0'], requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(params['beta'], requires_grad=True))
        
        self.A_CIL = nn.Parameter(torch.tensor(params['A_CIL'], requires_grad=True))
        
        self.A_CFs = torch.reshape(torch.tensor(params['A_CFs'], requires_grad=True), (-1,1))
        self.A_chems = torch.reshape(torch.tensor(params['A_chems'], requires_grad=True), (-1,1))

        self.A_ext = nn.Parameter(torch.tensor(params['A_ext'], requires_grad=True))

        self.celltypeModules()

        self.prepare_paramList()

        self.flg_periodic = not(periodic is None)
        
        if self.flg_periodic:
            self.periodic = torch.tensor(periodic)
        else:
            self.periodic = periodic
            
        self.def_dr()

        
        self.positionName = ut.variableInitializer(positionName, 'x')
        self.velocityName = ut.variableInitializer(velocityName, 'v')
        self.polarityName = ut.variableInitializer(polarityName, 'theta')
        self.torqueName = ut.variableInitializer(torqueName, 'w')
        self.noiseName = ut.variableInitializer(noiseName, 'sigma')
        
        self.celltypeName = ut.variableInitializer(celltypeName, 'celltype')
        
        self.torquemessageName = ut.variableInitializer(torquemessageName, 'm_t')
        self.velocitymessageName = ut.variableInitializer(velocitymessageName, 'm_v')

    def celltypeModules(self):
        self.A_CFs_module = nn.Embedding(self.N_celltypes, 1, _weight=self.A_CFs)
        self.A_chem_module = nn.Embedding(self.N_celltypes, 1, _weight=self.A_chem)        
    
    def prepare_paramList(self):
        self.paramList = {'kappa': self.J_chem.kappa,
                          'cutoff': self.J_chem.cutoff,
                          'r': self.J_CIL.r,
                          'u0': self.u0,
                          'beta': self.beta,
                          'A_CIL': self.A_CIL,
                          'A_ext': self.A_ext }
    
    def reset_param_func(self, target, value):
        if value is None:
            nn.init.uniform_(target)
        else:
            nn.init.constant_(target, value)
        
    def reset_parameter(self, params={}, sigma=None, A_CFs=None, A_chems=None):
        for key in params.keys():
            self.reset_param_func(self.paramList[key], params[key])

        self.J_chem.set_k1()
        
        if not A_CFs is None:
            self.A_CFs = torch.reshape(torch.tensor(params['A_CFs'], requires_grad=True), (-1,1))
        if not A_chems is None:
            self.A_chems = torch.reshape(torch.tensor(params['A_chems'], requires_grad=True), (-1,1))

        self.celltypeModules()
        
        self.reset_param_func(self.sigma, sigma)

        self.prepare_sigma()

        self.prepare_paramList()
        
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([self.N_dim,self.N_dim-1], device=self.sigma.device), self.sigma*torch.eye(self.N_dim-1, device=self.sigma.device)), dim=0)
        
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
        p_neighbor = self.polarity2vector(self, edges.src[self.polarityName])
        p_target = self.polarity2vector(self, edges.dst[self.polarityName])
        
        dr = self.distanceCalc(edges.dst[self.positionName], edges.src[self.positionName])
        abs_dr = torch.norm(dr, dim=-1, keepdim=True)
        unit_dr = nn.functional.normalize(dr, dim=-1)

        drp_inner = torch.sum(unit_dr * p_neighbor, dim=-1, keepdim=True)
        drp_cross = unit_dr[..., :1] * p_target[..., 1:2] - unit_dr[..., 1:2] * p_target[..., :1]

        J_CIL = self.J_CIL(abs_dr)
        J_CF = self.J_CF(unit_dr, p_neighbor)
        J_chem = self.J_chem(abs_dr)

        A_CF = self.A_CFs_module(edges.dst[self.celltypeName])
        A_chem = self.A_chem_module(edges.dst[self.celltypeName])

        return {self.velocitymessageName: -self.beta*J_CIL*unit_dr,
                self.torquemessageName: (A_CF*J_CF - self.A_CIL*J_CIL)*drp_cross + A_chem*J_chem*drp_inner}
    
    def aggregate_message(self, nodes):
        return {self.velocityName: torch.mean(nodes.mailbox[self.velocitymessageName], 1),
                self.torqueName : torch.mean(nodes.mailbox[self.torquemessageName], 1) }
        
    def polarity2vector(self, theta):
        return torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1)
        
    def f(self, t, g, args=None):
        g.update_all(self.calc_message, self.aggregate_message)
        p = self.polarity2vector(g.ndata[self.polarityName])
        g.ndata[self.velocityName] = g.ndata[self.velocityName] + self.u0 * p
        g.ndata[self.torqueName] = g.ndata[self.torqueName] + self.A_ext * p[..., :1]
        return g
      
    def g(self, t, g, args=None):
        self.prepare_sigma()
        g.ndata[self.noiseName] = self.sigmaMatrix.repeat(g.ndata[self.positionName].shape[0], 1, 1).to(g.device)
        return g


class thetaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return 1 - torch.mean(torch.cos(x - y))

class myLoss(nn.Module):
    def __init__(self, distanceCalc, N_dim=2, useScore=True):
        super().__init__()
        
        self.distanceCalc = distanceCalc
        self.useScore = useScore
                
        self.xyLoss = nn.MSELoss()
        self.thetaLoss = thetaLoss()
        
        self.N_dim = N_dim
        
        self.def_forward()
        
    def forward_score(self, x, y, score_x, score_y):
        dxy = self.distanceCalc(x[...,:self.N_dim], y[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x[...,self.N_dim:(2*self.N_dim-1)], y[...,self.N_dim:(2*self.N_dim-1)])
        scoreLoss = torch.mean(torch.square(torch.sum(score_x, dim=-1, keepdim=True) - score_y))
        return xyLoss, thetaLoss, scoreLoss
       
    def forward_noScore(self, x, y):
        dxy = self.distanceCalc(x[...,:self.N_dim], y[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x[...,self.N_dim:(2*self.N_dim-1)], y[...,self.N_dim:(2*self.N_dim-1)])
        return xyLoss, thetaLoss
       
    def def_forward(self):
        if self.useScore:
            self.forward = self.forward_score
        else:
            self.forward = self.forward_noScore
        


     
class myLoss_normalized(nn.Module):
    def __init__(self, distanceCalc, N_dim=2, useScore=True):
        super().__init__()
        
        self.distanceCalc = distanceCalc
        self.useScore = useScore
                
        self.xyLoss = nn.MSELoss()
        self.thetaLoss = thetaLoss()
        
        self.N_dim = N_dim
        
        self.def_forward()
        
    def forward_score(self, x_pred, x_truth, score_pred, score_truth):
        dxy = self.distanceCalc(x_pred[...,:self.N_dim], x_truth[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x_pred[...,self.N_dim:(2*self.N_dim-1)], x_truth[...,self.N_dim:(2*self.N_dim-1)])
        scoreLoss = torch.mean(torch.square(torch.sum(score_pred, dim=-1, keepdim=True) - score_truth))

        x_truth_zero = self.distanceCalc(torch.zeros_like(x_pred[...,:self.N_dim]), x_truth[...,:self.N_dim])
        xyNorm = self.xyLoss(x_truth_zero, torch.zeros_like(dxy))
        thetaNorm = self.thetaLoss(torch.zeros_like(x_pred[...,self.N_dim:(2*self.N_dim-1)]), x_truth[...,self.N_dim:(2*self.N_dim-1)])
        scoreNorm = torch.mean(torch.square(score_truth))
        return xyLoss/xyNorm, vLoss/vNorm, scoreLoss/scoreNorm
       
    def forward_noScore(self, x_pred, x_truth):
        dxy = self.distanceCalc(x_pred[...,:self.N_dim], x_truth[...,:self.N_dim])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x_pred[...,self.N_dim:(2*self.N_dim-1)], x_truth[...,self.N_dim:(2*self.N_dim-1)])

        x_truth_zero = self.distanceCalc(torch.zeros_like(x_pred[...,:self.N_dim]), x_truth[...,:self.N_dim])
        xyNorm = self.xyLoss(x_truth_zero, torch.zeros_like(dxy))
        thetaNorm = self.thetaLoss(torch.zeros_like(x_pred[...,self.N_dim:(2*self.N_dim-1)]), x_truth[...,self.N_dim:(2*self.N_dim-1)])
        return xyLoss/xyNorm, vLoss/vNorm
        
    def def_forward(self):
        if self.useScore:
            self.forward = self.forward_score
        else:
            self.forward = self.forward_noScore     





'''


class multitypedCMsimulate(mo.dynamicGNDEmodule):
    def __init__(self, params):
        super().__init__()

        self.J_chem = J_chemoattractant2D(params['kappa'], params['cutoff'])
        self.J_CF = J_contactFollowing()
        self.J_CIL = J_contactInhibitionOfLocomotion(params['r'])

        self.u0 = nn.Parameter(torch.tensor(params['u0'], requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(params['beta'], requires_grad=True))
        
        self.A_CFs = nn.Parameter(torch.tensor(params['A_CFs'], requires_grad=True))
        self.A_CIL = nn.Parameter(torch.tensor(params['A_CIL'], requires_grad=True))
        self.A_chem = nn.Parameter(torch.tensor(params['A_chem'], requires_grad=True))

        self.A_ext = nn.Parameter(torch.tensor(params['A_ext'], requires_grad=True))

        self.L = params['L']
        self.D = params['D']
        
    def forward(self, edges):
        dx = periodic_distance(edges.dst['x'], edges.src['x'], self.L)

        costheta = torch.cos(edges.dst['theta'])
        sintheta = torch.sin(edges.dst['theta'])

        dx_para = costheta * dx[..., :1] + sintheta * dx[..., 1:]
        dx_perp = costheta * dx[..., 1:] - sintheta * dx[..., :1]

        p_para_src = torch.cos(edges.src['theta'] - edges.dst['theta'])
        p_perp_src = torch.sin(edges.src['theta'] - edges.dst['theta'])

        rot_m_v = self.interactNN(torch.concat((dx_para, dx_perp, 
                                                p_para_src, p_perp_src,
                                                edges.dst['type'], edges.src['type']), -1))

        m_v = torch.concat((costheta * rot_m_v[..., :1] - sintheta * rot_m_v[..., 1:],
                            costheta * rot_m_v[..., 1:] + sintheta * rot_m_v[..., :1]), -1)

        m_theta = self.thetaDotNN(torch.concat((dx_para, dx_perp, 
                                                p_para_src, p_perp_src, 
                                                edges.dst['type'], edges.src['type']), -1))
        
        return {'m': torch.concat((m_v, m_theta), -1)}

    def f(self, t, y, gr, dynamicName, batchIDName):
        return None

    def g(self, t, y, gr, dynamicName, batchIDName):
        return None



'''





    
