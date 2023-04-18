import numpy as np
import torch
from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.wrapper_modules as wm
import collectiveMotionNN.graph_utils as gu




    


def bool2edge(boolMatrix):
    edges = torch.argwhere(boolMatrix)
    return (edges[:,0], edges[:,1])

def radiusGraphEdge_selfLoop(distanceMatrix, r0):
    return distanceMatrix < r0

def radiusGraphEdge_noSelfLoop(distanceMatrix, r0):
    distanceMatrix.fill_diagonal_(r0+1)
    return distanceMatrix < r0

class distance2edge_selfLoop(nn.Module):
    def __init__(self, r0):
        super().__init__()
        self.r0 = r0
        
    def forward(self, distanceMatrix):
        boolMatrix = radiusGraphEdge_selfLoop(distanceMatrix, self.r0)
        return bool2edge(boolMatrix)
    
class distance2edge_noSelfLoop(nn.Module):
    def __init__(self, r0):
        super().__init__()
        self.r0 = r0
        
    def forward(self, distanceMatrix):
        boolMatrix = radiusGraphEdge_noSelfLoop(distanceMatrix, self.r0)
        return bool2edge(boolMatrix)

    
    
def bool2edge_batched(boolVector, edgeCands):
    eC = edgeCands[boolVector]
    return (eC[:,0], eC[:,1])

def radiusGraphEdge_batched(distanceVector, r0):
    return distanceVector < r0

class distance2edge_batched(nn.Module):
    def __init__(self, r0):
        super().__init__()
        self.r0 = r0
        
    def forward(self, input):
        distanceVector, edgeCands = input
        boolVector = radiusGraphEdge_batched(distanceVector, self.r0)
        return bool2edge_batched(boolVector, edgeCands)

    
        
class distanceSigmoid(nn.Module):
    def __init__(self, r_scale, selfloop, multiBatch):
        super().__init__()
        
        self.r_scale = r_scale
        self.multiBatch = multiBatch
        
        self.def_selfloop_batch(selfloop)
        
    def def_selfloop_batch(self, selfloop=None, multiBatch=None):
        if not(selfloop is None):
            self.selfloop = selfloop
        if not(multiBatch is None):
            self.multiBatch = multiBatch
        self.def_triu()
        
    def def_triu(self):
        if self.selfloop:
            self.triu = lambda x: torch.triu(x)
        else:
            self.triu = lambda x: torch.triu(x, diagonal=1)
        if self.multiBatch:
            self.forward = self.forward_batched
        else:
            self.forward = self.forward_nonBatched

        
    def forward_nonBatched(self, dr):
        dr0 = self.triu(dr/self.r_scale)
        return torch.stack((self.triu(torch.sigmoid(dr0)).reshape(-1), dr0.reshape(-1)), dim=1) # probability score and logit 
    
    def forward_batched(self, dr):
        dr0 = dr / self.r_scale
        return torch.stack((torch.sigmoid(dr0).reshape(-1), dr0.reshape(-1)), dim=1) # probability score and logit 
    
    
        
class radiusgraphEdge(wm.edgeScoreCalculationModule):
    def __init__(self, r0, periodicLength=None, selfLoop=False, variableName=None, returnScore=False, r1=None, scoreCalcModule=None, eps=None, multiBatch=False):
        super().__init__(returnScore, multiBatch)
           
        self.r0 = r0

        self.periodicLength = periodicLength
        
        self.selfLoop = selfLoop

        self.edgeVariable = ut.variableInitializer(variableName, 'x')

        r1 = ut.variableInitializer(r1, r0/10.0)
        
        self.scoreCalcModule = ut.variableInitializer(scoreCalcModule, distanceSigmoid(r1, self.selfLoop, self.multiBatch))
        
        self.eps = ut.variableInitializer(eps, 1e-5)
        
        self.def_dr()
        
        self.def_distance2edge()
        
        self.reset_multiBatch()
        
        
    def def_nonPeriodic(self):
        self.distanceCalc = ut.euclidDistance_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = ut.euclidDistance_periodic(self.periodicLength)
        
    def def_dr(self):
        if self.periodicLength is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic()
        

    def norm_dr(self, dr):
        flg_nz = torch.logical_or(dr > self.eps, dr < -self.eps)
        #for i in range(dr.shape[-1]):
        #    dr[:,:,i].fill_diagonal_(self.eps)
        dr = torch.norm(torch.where(flg_nz, dr, self.eps), dim=-1, keepdim=False)
        #dr.fill_diagonal_(0.0)
        return dr        
    
    def calc_abs_distance_nonBatch(self, g, args=None):
        dr = self.distanceCalc(torch.unsqueeze(g.ndata[self.edgeVariable], 0), torch.unsqueeze(g.ndata[self.edgeVariable], 1))
        return self.norm_dr(dr), None

    def calc_abs_distance_batch(self, bg, args=None):
        edgeCands, _ = self.edgeCandsCalc(bg)
        dr = self.distanceCalc(bg.ndata[self.edgeVariable][edgeCands[:,0]], bg.ndata[self.edgeVariable][edgeCands[:,1]])
        return self.norm_dr(dr), edgeCands
    
    def pass_dr_nonBatch(self, dr):
        return dr[0]
    
    def pass_dr_batch(self, dr):
        return dr
    
    def def_noSelfLoop(self):
        self.edgeCandsCalc = lambda bg: gu.sameBatchEdgeCandidateNodePairs_selfloop(bg)
        if not self.multiBatch:
            self.distance2edge = distance2edge_noSelfLoop(self.r0)
            self.calc_abs_distance = self.calc_abs_distance_nonBatch
            self.pass_dr = self.pass_dr_nonBatch
        
    def def_selfLoop(self):
        self.edgeCandsCalc = lambda bg: gu.sameBatchEdgeCandidateNodePairs_noSelfloop(bg)
        if not self.multiBatch:
            self.distance2edge = distance2edge_selfLoop(self.r0)
            self.calc_abs_distance = self.calc_abs_distance_nonBatch
            self.pass_dr = self.pass_dr_nonBatch
        
    def def_batched(self):
        self.distance2edge = distance2edge_batched(self.r0)
        self.calc_abs_distance = self.calc_abs_distance_batch
        self.pass_dr = self.pass_dr_batch
        
    def def_distance2edge(self):
        if self.selfLoop:
            self.def_selfLoop()
        else:
            self.def_noSelfLoop()
        if self.multiBatch:
            self.def_batched()
    
    
    def reset_multiBatch(self):
        if 'selfLoop' in vars(self).keys():
            self.def_distance2edge()
        
    
    def forward_noScore(self, g, args=None):
        out = self.calc_abs_distance(g, args)
        return self.distance2edge(self.pass_dr(out))

    def forward_score(self, g, args=None):
        out = self.calc_abs_distance(g, args)
        return self.distance2edge(self.pass_dr(out)), self.scoreCalcModule(self.r0 - out[0])
    
    
    
class pAndLogit2KLdiv(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1):
        return torch.tensor(list(map(lambda x: torch.mean(torch.prod(x[0] - x[1], dim=1)), zip(x0,x1))), device=x0[0].device)
    
class scoreListModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, postScore, preScore):
        preScore.append(postScore)
        return preScore
    
class scoreSumModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, postScore, preScore):
        return postScore + preScore
    
