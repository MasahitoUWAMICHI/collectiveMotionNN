import torch
import dgl

def update_edges(g, edges):
    g.remove_edges(g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges[0], edges[1])
    return g

def update_adjacency(g, edgeConditionFunc):
    edges = edgeConditionFunc(g)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeConditionFunc):
    gs = dgl.unbatch(bg)
    for g in gs:
        update_adjacency(g, edgeConditionFunc)
    bg = dgl.batch(gs)
    return bg

def judge_skipUpdate(g, dynamicVariable, dynamicName):
    return torch.allclose(g.ndata[dynamicName], dynamicVariable)


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
