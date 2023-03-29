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

def judge_skipUpdate(g, dynamicVariable, dynamicName):
    return torch.allclose(g.ndata[dynamicName], dynamicVariable)

def edgeRefresh_execute(gr, dynamicVariable, dynamicName, edgeCondtionModule, args=None):
    gr.ndata[dynamicName] = dynamicVariable
    gr = update_adjacency_batch(gr, edgeCondtionModule, args)
    return gr


class edgeRefresh_noForceUpdate(nn.Module):
    def __init__(self, edgeConditionModule):
        super().__init__()
        
        self.edgeConditionModule = edgeConditionModule
        
    def forceUpdate(self, gr, dynamicVariable, dynamicName, args=None):
        return edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule, args)
    
    def forward(self, gr, dynamicVariable, dynamicName, args=None):
        if judge_skipUpdate(gr, dynamicVariable, dynamicName):
            return gr
        else:
            return edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule, args)

        
class edgeRefresh_forceUpdate(nn.Module):
    def __init__(self, edgeConditionModule):
        super().__init__()
        
        self.edgeConditionModule = edgeConditionModule
        
    def forceUpdate(self, gr, dynamicVariable, dynamicName, args=None):
        return edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule, args)
    
    def forward(self, gr, dynamicVariable, dynamicName, args=None):
        return edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule, args)




    

def make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName):
    Nnodes = dynamicVariable.shape[0]
    g = dgl.graph((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)), num_nodes=Nnodes)
    g.ndata[dynamicName] = dynamicVariable

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

