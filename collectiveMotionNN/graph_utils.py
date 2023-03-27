import torch
import dgl

def update_edges(g, edges):
    g.remove_edges(g, g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges)
    return None

def calc_adjacency(g, edgeConditionFunc):
    adj = edgeConditionFunc(g.ndata)
    return torch.argwhere(adj)

def update_adjacency(g, edgeConditionFunc):
    edges = calc_adjacency(g, edgeConditionFunc)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeConditionFunc):
    gs = list(dgl.unbatch(bg))
    for i in range(len(gs)):
        gs[i] = update_adjacency(gs[i], edgeConditionFunc)
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

def make_disconnectedGraph_func_batch(dynamicVariable, staticVariables, dynamicName, batchFlg):
    Nnodes = torch.sum(batchFlg)
    g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=Nnodes)
    g.ndata[dynamicName] = dynamicVariable[batchFlg]

    for key in staticVariables.keys():
        g.ndata[key] = staticVariables[key][batchFlg]

    return g


def make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName='y', batchID=None):
    if batchID is None:
        g = make_disconnectedGraph_func(dynamicVariable, staticVariables, dynamicName)
        isBatched = False
    else:
        batchIDs = torch.unique(batchID)
        g = dgl.batch([make_disconnectedGraph_func_batch(dynamicVariable, staticVariables, dynamicName, batchID==bID)\
                        for bID in batchIDs])
        isBatched = True

    return g, dynamicName, isBatched
