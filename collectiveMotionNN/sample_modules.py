import numpy as np
import torch
from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.wrapper_modules as wm



class singleVariableNdataInOut(nn.Module):
    def __init__(self, variableName):
        super().__init__()
        
        self.variableName = variableName
    
    def input(self, gr, variableValue):
        gr.ndata[self.variableName] = variableValue
        return gr

    def output(self, gr):
        return gr.ndata[self.variableName]
    
class multiVariableNdataInOut(nn.Module):
    def __init__(self, variableName, variableNDims):
        super().__init__()
        
        assert len(variableName) == len(variableNDims)
        
        self.variableName = variableName
        self.variableNDims = variableNDims
        
        self.initializeIndices()
        
    def initializeIndices(self):
        self.variableIndices = np.cumsum(np.array([0]+list(self.variableNDims), dtype=int))
    
    def input(self, gr, variableValue):
        for vN, vD0, vD1 in zip(self.variableName, self.variableIndices[:-1], self.variableIndices[1:]):
            gr.ndata[vN] = variableValue[..., vD0:vD1]
        return gr

    def output(self, gr):
        return torch.cat([gr.ndata[vN] for vN in self.variableName], dim=-1)
    
    
    
    
    
    


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

    
        
class distanceSigmoid(nn.Module):
    def __init__(self, r_scale, selfloop):
        super().__init__()
        
        self.r_scale = r_scale
        
        self.def_selfloop(selfloop)
        
    def def_selfloop(self, selfloop)
        self.selfloop = selfloop
        if selfloop:
            self.triu = lambda x: torch.triu(x)
        else:
            self.triu = lambda x: torch.triu(x, diagonal=1)
        
    def forward(self, dr):
        dr0 = self.triu(dr/self.r_scale)
        return torch.stack((self.triu(torch.sigmoid(dr0)).reshape(-1), dr0.reshape(-1)), dim=1) # probability score and logit 
        
class pAndLogit2KLdiv(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1):
        return torch.sum(torch.prod(x0 - x1, dim=1))
        
    
class radiusgraphEdge(wm.edgeScoreCalculationModule):
    def __init__(self, r0, r1=None, periodicLength=None, selfLoop=False, variableName=None, returnScore=False, scoreCalcModule=None):
        super().__init__(returnScore)
           
        self.r0 = r0

        self.periodicLength = periodicLength
        
        self.selfLoop = selfLoop
        
        self.edgeVariable = ut.variableInitializer(variableName, 'x')

        self.r1 = ut.variableInitializer(r1, r0/10.0)
        
        self.scoreCalcModule = ut.variableInitializer(scoreCalcModule, distanceSigmoid(self.r1, selfloop))
        
        self.def_dr()
        
        self.def_distance2edge()
        
        
    def def_nonPeriodic(self):
        self.distanceCalc = ut.euclidDistance_nonPeriodic()
        
    def def_periodic(self):
        self.distanceCalc = ut.euclidDistance_periodic(self.periodicLength)
        
    def def_dr(self):
        if self.periodicLength is None:
            self.def_nonPeriodic()
        else:
            self.def_periodic()
        

    def def_noSelfLoop(self):
        self.distance2edge = distance2edge_noSelfLoop(self.r0)
        
    def def_selfLoop(self):
        self.distance2edge = distance2edge_selfLoop(self.r0)
        
    def def_distance2edge(self):
        if self.selfLoop:
            self.def_selfLoop()
        else:
            self.def_noSelfLoop()
    
    
    def calc_abs_distance(self, g, args=None):
        dr = self.distanceCalc(torch.unsqueeze(g.ndata[self.edgeVariable], 0), torch.unsqueeze(g.ndata[self.edgeVariable], 1))
        return torch.norm(dr, dim=-1, keepdim=False)
        
    def forward_noScore(self, g, args=None):
        dr = self.calc_abs_distance(g, args)
        return self.distance2edge(dr)

    def forward_score(self, g, args=None):
        dr = self.calc_abs_distance(g, args)
        return self.distance2edge(dr), self.calc_score(self.r0 - dr)
    
    
