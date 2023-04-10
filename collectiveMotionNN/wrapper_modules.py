from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu







class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInOutModule=None, derivativeInOutModule=None, args=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
        self.ndataInOutModule = ut.variableInitializer(ndataInOutModule, gu.singleVariableNdataInOut('x'))
            
        self.derivativeInOutModule = ut.variableInitializer(derivativeInOutModule, gu.singleVariableNdataInOut('v'))

        self.edgeInitialize(args)
        
        
    def edgeInitialize(self, args=None):
        self.graph = self.dynamicGNDEmodule.edgeInitialize(self.graph, args)

    def loadGraph(self, graph, args=None):
        self.graph = graph
        self.edgeInitialize(args)
        
    def loadNdata_noEdgeInitialize(self, x):
        self.graph = self.ndataInOutModule.input(self.graph, x)
        
    def loadNdata_edgeInitialize(self, x, args=None):
        self.loadNdata_noEdgeInitialize(x)
        self.edgeInitialize(args)
        
    def deleteGraph(self):
        self.graph = None
        self.dynamicGNDEmodule.edgeRefresher.graph = None
        
    def forward(self, t, x, args=None):
        self.graph = self.dynamicGNDEmodule.f(t, x, self.graph, self.ndataInOutModule, args)
        return self.derivativeInOutModule.output(self.graph)
    

class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInOutModule=None, derivativeInOutModule=None, noiseInOutModule=None, noise_type=None, sde_type=None, args=None):
        super().__init__(dynamicGNDEmodule, graph, ndataInOutModule, derivativeInOutModule, args)
        
        self.noiseInOutModule = ut.variableInitializer(noiseInOutModule, gu.singleVariableNdataInOut('sigma'))

        self.noise_type = ut.variableInitializer(noise_type, 'general')
        self.sde_type = ut.variableInitializer(sde_type, 'ito')
        
    def f(self, t, x, args=None):
        return self.forward(t, x, args)
    
    def g(self, t, x, args=None):
        self.graph = self.dynamicGNDEmodule.g(t, x, self.graph, self.ndataInOutModule, args)
        return self.noiseInOutModule.output(self.graph)



    
    
    
    
class dynamicGNDEmodule(nn.Module):
    def __init__(self, calc_module, edgeConditionModule, forceUpdate=None, rtol=None, atol=None, equal_nan=None, returnScore=None, scorePostProcessModule=None, scoreIntegrationModule=None):
        super().__init__()
        
        self.calc_module = calc_module

        #self.edgeConditionModule = edgeConditionModule
        
        self.forceUpdate = ut.variableInitializer(forceUpdate, False)
        
        self.edgeRefresher = gu.edgeRefresh(self.edgeConditionModule, returnScore, scorePostProcessModule, scoreIntegrationModule, forceUpdate, rtol, atol, equal_nan)
        
    def edgeInitialize(self, gr, args=None):
        return self.edgeRefresher.createEdge(gr, args)

    def f(self, t, x, gr, ndataInOutModule, args=None):
        gr = self.edgeRefresher(t, gr, x, ndataInOutModule, args)
        return self.calc_module.f(t, gr, args)

    def g(self, t, x, gr, ndataInOutModule, args=None):
        gr = self.edgeRefresher(t, gr, x, ndataInOutModule, args)
        return self.calc_module.g(t, gr, args)

    
    
class edgeScoreCalculationModule(nn.Module):
    def __init__(self, returnScore=False):
        super().__init__()
        self.set_returnScore(returnScore)
        
    def set_returnScore(self, returnScore):
        self.returnScore = returnScore
        if self.returnScore:
            self.forward = self.forward_score
        else:
            self.forward = self.forward_noScore
            
    def forward_score(self, x):
        return None
        
    def forward_noScore(self, x):
        return None
   

