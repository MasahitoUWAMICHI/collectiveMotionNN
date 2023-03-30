import numpy as np
import torch
from torch import nn
import dgl

def update_edges(g, edges):
    g.remove_edges(g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges[0], edges[1])
    return g

def update_adjacency(g, edgeCondtionModule, args=None):
    edges = edgeCondtionModule(g, args)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeCondtionModule, args=None):
    gs = dgl.unbatch(bg)
    for g in gs:
        update_adjacency(g, edgeCondtionModule, args)
    bg = dgl.batch(gs)
    return bg

def judge_skipUpdate(g, dynamicVariable, ndataInOutModule):
    return torch.allclose(ndataInOutModule.output(g), dynamicVariable)

def edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, edgeCondtionModule, args=None):
    gr = ndataInOutModule.input(gr, dynamicVariable)
    gr = update_adjacency_batch(gr, edgeCondtionModule, args)
    return gr


class edgeRefresh_noForceUpdate(nn.Module):
    def __init__(self, edgeConditionModule):
        super().__init__()
        
        self.edgeConditionModule = edgeConditionModule
        
    def createEdge(self, gr, args=None):
        return update_adjacency_batch(gr, self.edgeCondtionModule, args)
    
    def forward(self, gr, dynamicVariable, ndataInOutModule, args=None):
        if judge_skipUpdate(gr, dynamicVariable, ndataInOutModule):
            return gr
        else:
            return edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.edgeConditionModule, args)

        
class edgeRefresh_forceUpdate(edgeRefresh_noForceUpdate):
    def __init__(self, edgeConditionModule):
        super().__init__(edgeConditionModule)
            
    def forward(self, gr, dynamicVariable, ndataInOutModule, args=None):
        return edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.edgeConditionModule, args)




    
class singleVariableNdataInOut(nn.Module):
    def __init__(self, variableName):
        super().__init__()
        
        self.variableName = variableName
    
    def input(self, gr, variableValue):
        gr[self.variableName] = variableValue
        return gr

    def output(self, gr):
        return gr[self.variableName]
    
class multiVariableNdataInOut(nn.Module):
    def __init__(self, variableName, variableDims):
        super().__init__()
        
        assert len(variableName) == len(variableDims)
        
        self.variableName = variableName
        self.variableDims = variableDims
        
        self.initializeIndices()
        
    def initializeIndices(self):
        self.variableIndices = np.cumsum(np.array([0]+list(self.variableDims), dtype=int))
    
    def input(self, gr, variableValue):
        for vN, vD0, vD1 in zip(self.variableName, self.variableIndices[:-1], self.variableIndices[1:]):
            gr[vN] = variableValue[..., vD0:vD1]
        return gr

    def output(self, gr):
        return torch.cat([gr[vN] for vN in self.variableName], dim=-1)
    
    
    
    

def make_disconnectedGraph(dynamicVariable, staticVariables, ndataInOutModule):
    Nnodes = dynamicVariable.shape[0]
    g = dgl.graph((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)), num_nodes=Nnodes)
    g = ndataInOutModule.input(g, dynamicVariable)

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

