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

def judge_skipUpdate(g, dynamicVariable, ndataInOutModule, rtol=1e-05, atol=1e-08, equal_nan=True):
    return torch.allclose(ndataInOutModule.output(g), dynamicVariable, rtol, atol, equal_nan)

def edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, edgeConditionModule, args=None):
    gr = ndataInOutModule.input(gr, dynamicVariable)
    gr = update_adjacency_batch(gr, edgeConditionModule, args)
    return gr


        
class edgeRefresh_forceUpdate(nn.Module):
    def __init__(self, edgeConditionModule):
        super().__init__()

        self.edgeConditionModule = edgeConditionModule

    def createEdge(self, gr, args=None):
        return update_adjacency_batch(gr, self.edgeConditionModule, args)
    
    def forward(self, gr, dynamicVariable, ndataInOutModule, args=None):
        return edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.edgeConditionModule, args)


class edgeRefresh_noForceUpdate(edgeRefresh_forceUpdate):
    def __init__(self, edgeConditionModule, rtol=None, atol=None, equal_nan=None):
        super().__init__(edgeConditionModule)
        
        self.rtol = ut.variableInitializer(rtol, 1e-05)
        self.atol = ut.variableInitializer(atol, 1e-08)
        self.equal_nan = ut.variableInitializer(equal_nan, True)
        
    def forward(self, gr, dynamicVariable, ndataInOutModule, args=None):
        if judge_skipUpdate(gr, dynamicVariable, ndataInOutModule, self.rtol, self.atol, self.equal_nan):
            return gr
        else:
            return edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.edgeConditionModule, args)




    
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

