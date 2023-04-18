import numpy as np
import torch
from torch import nn
import dgl

import collectiveMotionNN.utils as ut

def nodeIDrange_eachBatch(bg):
    eachBatchNodeID_end = torch.cumsum(bg.batch_num_nodes(), 0)
    return torch.stack((eachBatchNodeID_end - bg.batch_num_nodes(), eachBatchNodeID_end), dim=1)

def IDrange2MatrixID(IDrange):
    x = torch.arange(IDrange[0], IDrange[1], device=IDrange.device)
    return torch.cartesian_prod(x, x)

def isSelfloop(matrixID):
    return matrixID[:,0]==matrixID[:,1]

def removeSelfloop(matrixID, flgSelfloop):
    return matrixID[torch.logical_not(flgSelfloop)]

def matrixID_flgSelfloop(IDrange):
    xx = IDrange2MatrixID(IDrange)
    return xx, isSelfloop(xx)

def sameBatchEdgeCandidateNodePairs_selfloop(bg):
    edgeCandidatesAndSelfloops = list(map(matrixID_flgSelfloop, nodeIDrange_eachBatch(bg)))
    edgeCandidates, selfloops = list(zip(*edgeCandidatesAndSelfloops))
    return torch.cat(edgeCandidates, dim=0), torch.cat(selfloops, dim=0)

def sameBatchEdgeCandidateNodePairs_noSelfloop(bg):
    edge, selfloops = sameBatchEdgeCandidateNodePairs_selfloop(bg)
    edge = removeSelfloop(edge, selfloops)
    return edge, torch.full([edge.shape[0]], False)


def edge2batchNumEdges(edges, bnn):
    nodeID_ends = torch.cumsum(bnn, 0)
    edge_batchIDs = torch.count_nonzero(torch.unsqueeze(edges[0], 1) >= torch.unsqueeze(nodeID_ends, 0), dim=1)
    return edge_batchIDs.bincount(minlength=len(bnn))


def update_edges(g, edges):
    bnn = g.batch_num_nodes().clone()
    bne = edge2batchNumEdges(edges, bnn)
    g.remove_edges(g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges[0].to(g.device), edges[1].to(g.device))
    g.set_batch_num_nodes(bnn)
    g.set_batch_num_edges(bne)
    return g


def update_adjacency(g, edgeConditionModule, args=None):
    edges = edgeConditionModule(g, args)
    update_edges(g, edges)
    return g

def update_adjacency_batch(bg, edgeConditionModule, unbatchFunc, args=None):
    gs = list(map(lambda g: update_adjacency(g, edgeConditionModule, args), unbatchFunc(bg)))
    bg = dgl.batch(gs)
    return bg, None


def update_adjacency_returnScore(g, edgeConditionModule, args=None):
    edges, score = edgeConditionModule(g, args)
    update_edges(g, edges)
    return g, score

def update_adjacency_returnScore_batch(bg, edgeConditionModule, unbatchFunc, args=None):
    gscore = list(map(lambda g: list(update_adjacency_returnScore(g, edgeConditionModule, args)), unbatchFunc(bg))) # list of lists [graph, score]
    gs, scores = list(zip(*gscore))
    bg = dgl.batch(gs)
    return bg, scores

def make_multiBatches(bg, N_multiBatch):
    gs = dgl.unbatch(bg)
    multiBatch_cuts = np.append(np.arange(0, len(gs), N_multiBatch), len(gs))
    return list(map(lambda x: dgl.batch(gs[x[0]:x[1]]).to(bg.device), zip(multiBatch_cuts[:-1], multiBatch_cuts[1:])))

def judge_skipUpdate(g, dynamicVariable, ndataInOutModule, rtol=1e-05, atol=1e-08, equal_nan=True):
    return torch.allclose(ndataInOutModule.output(g), dynamicVariable, rtol, atol, equal_nan)

def edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, updateFunc, args=None):
    gr = ndataInOutModule.input(gr, dynamicVariable)
    return updateFunc(gr, args)

    
class edgeRefresh(nn.Module):
    def __init__(self, edgeConditionModule, returnScore=None, scorePostProcessModule=None, scoreIntegrationModule=None, forceUpdate=None, N_multiBatch=None, rtol=None, atol=None, equal_nan=None):
        super().__init__()

        self.edgeConditionModule = edgeConditionModule
        
        self.rtol = ut.variableInitializer(rtol, 1e-05)
        self.atol = ut.variableInitializer(atol, 1e-08)
        self.equal_nan = ut.variableInitializer(equal_nan, True)
        
        self.N_multiBatch = ut.variableInitializer(N_multiBatch, 1)
        self.returnScore = ut.variableInitializer(returnScore, False)
        self.forceUpdate = ut.variableInitializer(forceUpdate, False)
        self.scorePostProcessModule = scorePostProcessModule
        self.scoreIntegrationModule = scoreIntegrationModule
        
        self.def_unbatch()
        
        self.def_graph_updates()
        
        self.def_forward()
        
        self.resetScores()

        
    def unbatch_single(self, bg):
        return dgl.unbatch(bg)
    
    def unbatch_multi(self, bg):
        return make_multiBatches(bg, self.N_multiBatch)
    
    def def_unbatch(self):
        if self.N_multiBatch > 1:
            self.unbatch = self.unbatch_multi
        else:
            self.unbatch = self.unbatch_single
            
    def reset_N_multiBatch(self, N_multiBatch):
        self.N_multiBatch = N_multiBatch
        self.def_unbatch()
        self.edgeConditionModule.set_multiBatch(N_multiBatch > 1)
        
    def update_adjacency_batch(self, gr, args=None):
        return update_adjacency_batch(gr, self.edgeConditionModule, self.unbatch, args)

    def update_adjacency_returnScore_batch(self, gr, args=None):
        return update_adjacency_returnScore_batch(gr, self.edgeConditionModule, self.unbatch, args)
    
    def def_noScore(self):
        self.update_adjacency = self.update_adjacency_batch
        self.postProcess = lambda x, flg=None: x[0]
        
    def def_score(self):
        self.update_adjacency = self.update_adjacency_returnScore_batch
        self.postProcess = self.postProcess_score
        
    def def_graph_updates(self):
        if self.returnScore:
            self.def_score()
        else:
            self.def_noScore()
            
    def reset_returnScoreMode(self, returnScore):
        self.returnScore = returnScore
        self.def_graph_updates()
        self.edgeConditionModule.set_returnScore(returnScore)
        
    
    def def_forward(self):
        if self.forceUpdate:
            self.forward = self.forward_forceUpdate
        else:
            self.forward = self.forward_noForceUpdate
            
    def reset_forceUpdateMode(self, forceUpdate):
        self.forceUpdate = forceUpdate
        self.def_forward()
    
    
    def loadGraph(self, gr):
        self.graph = gr
        
    def loadScore(self, score):
        self.score = score
                
    def loadProcessedScore(self, ps):
        self.processedScore = ps
        
    def loadTimeStamp(self, t):
        self.lastScoreCalculationTime = t
        
    def createEdge(self, gr, score=None, ps=None, t=None, args=None):
        self.loadGraph(gr)
        out = self.update_adjacency(gr, args)
        self.resetScores(ut.variableInitializer(score, out[1]), ps, t)
        return out[0]
    
    def postProcess_score(self, out, t):
        if t > self.lastScoreCalculationTime:
            self.resetScores(score = out[1],
                             ps = self.scoreIntegrationModule(self.scorePostProcessModule(self.score, out[1]), self.processedScore),
                             t = t)
        return out[0]
    
    def resetScores(self, score=None, ps=None, t=None):
        self.loadScore(score)
        self.loadProcessedScore(ut.variableInitializer(ps, []))
        self.loadTimeStamp(ut.variableInitializer(t, 0))
        
    def deleteGraphs(self):
        self.graph = None
        self.resetScores()
    
    def forward_forceUpdate(self, t, gr, dynamicVariable, ndataInOutModule, args=None):
        out = edgeRefresh_execute(gr, dynamicVariable, ndataInOutModule, self.update_adjacency, args)
        gr = self.postProcess(out, t)
        self.loadGraph(gr)
        return gr

    def forward_noForceUpdate(self, t, gr, dynamicVariable, ndataInOutModule, args=None):
        if judge_skipUpdate(self.graph, dynamicVariable, ndataInOutModule, self.rtol, self.atol, self.equal_nan):
            return gr
        else:
            gr = self.forward_forceUpdate(t, gr, dynamicVariable, ndataInOutModule, args)
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





    
