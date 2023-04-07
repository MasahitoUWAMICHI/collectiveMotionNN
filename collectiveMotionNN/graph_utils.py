import numpy as np
import torch
from torch import nn
import dgl

import collectiveMotionNN.utils as ut

def update_edges(g, edges):
    g.remove_edges(g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges[0], edges[1])
    return g


def update_adjacency(g, edgeConditionModule, args=None):
    edges = edgeConditionModule(g, args)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeConditionModule, args=None):
    gs = dgl.unbatch(bg)
    for g in gs:
        update_adjacency(g, edgeConditionModule, args)
    bg = dgl.batch(gs)
    return bg


def update_adjacency_returnScore(g, edgeConditionModule, args=None):
    edges, score = edgeConditionModule(g, args)
    update_edges(g, edges)
    return g, score

def update_adjacency_returnScore_batch(bg, edgeConditionModule, args=None):
    gs = dgl.unbatch(bg)
    scores = []
    for i, g in enumerate(gs):
        gs[i], score = update_adjacency_returnScore(g, edgeConditionModule, args)
        scores.append(score)
    bg = dgl.batch(gs)
    return bg, scores


def judge_skipUpdate(g, dynamicVariable, ndataInOutModule, rtol=1e-05, atol=1e-08, equal_nan=True):
    return torch.allclose(ndataInOutModule.output(g), dynamicVariable, rtol, atol, equal_nan)

def edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, edgeConditionModule, updateFunc, args=None):
    gr = ndataInOutModule.input(gr, dynamicVariable)
    return updateFunc(gr, edgeConditionModule, args)

    
class edgeRefresh_forceUpdate(nn.Module):
    def __init__(self, edgeConditionModule, returnScore=None, forceUpdate=None, rtol=None, atol=None, equal_nan=None):
        super().__init__()

        self.edgeConditionModule = edgeConditionModule
        
        self.rtol = ut.variableInitializer(rtol, 1e-05)
        self.atol = ut.variableInitializer(atol, 1e-08)
        self.equal_nan = ut.variableInitializer(equal_nan, True)
        
        self.returnScore = ut.variableInitializer(returnScore, False)
        self.forceUpdate = ut.variableInitializer(forceUpdate, False)
        
        self.def_graph_updates()
        
        self.def_forward()
        

    def def_noScore(self):
        self.update_adjacency = lambda gr, args=None: update_adjacency_batch(gr, self.edgeConditionModule, args)
        self.postProcess = lambda x: x
        
    def def_score(self):
        self.update_adjacency = lambda gr, args=None: update_adjacency_returnScore_batch(gr, self.edgeConditionModule, args)
        self.postProcess = self.postProcess_score
        
    def def_graph_updates(self):
        if self.returnScore:
            self.def_score()
        else:
            self.def_noScore()
            
    def reset_returnScoreMode(self, returnScore):
        self.returnScore = returnScore
        self.def_graph_updates()
        
    
    def def_forceUpdate(self):
        self.forward = self.forward_forceUpdate
        
    def def_noForceUpdate(self):
        self.forward = self.forward_noForceUpdate
        
    def def_forward(self):
        if self.forceUpdate:
            self.def_forceUpdate()
        else:
            self.def_noForceUpdate()
            
    def reset_forceUpdateMode(self, forceUpdate):
        self.forceUpdate = forceUpdate
        self.def_forward()
    
    
    def loadGraph(self, gr):
        self.graph = gr
        
    def createEdge(self, gr, args=None):
        self.loadGraph(gr)
        return self.update_adjacency(gr, args)
    
    def postProcess_score(self, out)
        
        return out[0]
    
    def forward_forceUpdate(self, gr, dynamicVariable, ndataInOutModule, args=None):
        out = edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.edgeConditionModule, self.update_adjacency, args)
        gr = self.postProcess(out)
        self.loadGraph(gr)
        return gr

    def forward_noForceUpdate(self, gr, dynamicVariable, ndataInOutModule, args=None):
        if judge_skipUpdate(self.graph, dynamicVariable, ndataInOutModule, self.rtol, self.atol, self.equal_nan):
            return gr
        else:
            gr = self.forward_forceUpdate(gr, dynamicVariable, ndataInOutModule, args)
            return gr






    
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
    
    
    
    

def make_disconnectedGraph(dynamicVariable, ndataInOutModule, staticVariables=None):
    Nnodes = dynamicVariable.shape[0]
    g = dgl.graph((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)), num_nodes=Nnodes)
    g = ndataInOutModule.input(g, dynamicVariable)
    
    staticVariables = ut.variableInitializer(staticVariables, {})

    for key in staticVariables.keys():
        g.ndata[key] = staticVariables[key]

    return g





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

    
    
class radiusgraphEdge(nn.Module):
    def __init__(self, r0, periodicLength=None, selfLoop=False, variableName=None):
        super().__init__()
           
        self.r0 = r0

        self.periodicLength = periodicLength
        
        self.selfLoop = selfLoop
        
        self.edgeVariable = ut.variableInitializer(variableName, 'x')
        
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
            
            
        
    def forward(self, g, args=None):
        dr = self.distanceCalc(torch.unsqueeze(g.ndata[self.edgeVariable], 0), torch.unsqueeze(g.ndata[self.edgeVariable], 1))
        dr = torch.norm(dr, dim=-1, keepdim=False)
        return self.distance2edge(dr)            
    
    
