import torch
import dgl

def update_edges(g, edges):
    g.remove_edges(g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges[0], edges[1])
    return g

def calc_adjacency(g, edgeConditionFunc):
    adj = edgeConditionFunc(g.ndata)
    edges = torch.argwhere(adj)
    return (edges[:,0], edges[:,1])

def update_adjacency(g, edgeConditionFunc):
    edges = calc_adjacency(g, edgeConditionFunc)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeConditionFunc):
    gs = list(dgl.unbatch(bg))
    for g in gs:
        update_adjacency(g, edgeConditionFunc)
    bg = dgl.batch(gs)
    return bg

def judge_skipUpdate(g, dynamicVariable, dynamicName):
    return torch.allclose(g.ndata[dynamicName], dynamicVariable)


def make_disconnectedGraph_func(dynamicVariable, staticVariables, dynamicName):
    Nnodes = dynamicVariable.shape[0]
    g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=Nnodes)
    g.ndata[dynamicName] = dynamicVariable

    for key in staticVariables.keys():
        g.ndata[key] = staticVariables[key]

    return g
