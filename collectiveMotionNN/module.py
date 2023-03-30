from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu







class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInputModule=None, ndataOutputModule=None, velocityOutputModule=None, args=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
        self.ndataInputModule = ut.variableInitializer(ndataInputModule, gu.singleVariableNdataInput('y'))
            
        self.ndataOutputModule = ut.variableInitializer(ndataOutputModule, gu.singleVariableNdataOutput('y'))

        self.velocityOutputModule = ut.variableInitializer(velocityOutputModule, gu.singleVariableNdataOutput('v'))

        self.edgeInitialize(args)
        
    def loadGraph(self, dynamicVariable, staticVariables):
        self.graph = gu.make_disconnectedGraph(dynamicVariable, staticVariables, self.ndataInputModule)
    
    def edgeInitialize(self, args=None):
        self.graph = self.dynamicGNDEmodule.edgeInitialize(self.graph, self.ndataInputModule, args)

    def f(self, t, y, args=None):
        self.graph = self.dynamicGNDEmodule.f(t, y, self.graph, self.ndataInputModule, self.ndataOutputModule, args)
        return self.ndataOutputModule(self.graph)
    
    def forward(self, t, y, args=None):
        return self.f(t, y, args)


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInputModule=None, ndataOutputModule=None, velocityOutputModule=None, noiseOutputModule=None, args=None):
        super().__init__(dynamicGNDEmodule, graph, ndataInputModule, ndataOutputModule, velocityOutputModule, args)
        
        self.noiseOutputModule = ut.variableInitializer(noiseOutputModule, gu.singleVariableNdataOutput('s'))

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y, args=None):
        self.graph = self.dynamicGNDEmodule.g(t, y, self.graph, self.ndataInputModule, self.ndataOutputModule, args)
        return self.noiseOutputModule(self.graph)



    
    
    
    
class dynamicGNDEmodule(nn.Module):
    def __init__(self, calc_module, edgeConditionModule, forceUpdate=False):
        super().__init__()
        
        self.calc_module = calc_module

        self.edgeConditionModule = edgeConditionModule
        
        self.forceUpdate = forceUpdate

        self.def_edgeRefresher()
        
    def def_edgeRefresher_forceUpdate(self):
        self.edgeRefresher = gu.edgeRefresh_forceUpdate(self.edgeConditionModule)

    def def_edgeRefresher_noForceUpdate(self):
        self.edgeRefresher = gu.edgeRefresh_noForceUpdate(self.edgeConditionModule)
        
    def def_edgeRefresher(self):
        if self.forceUpdate:
            self.def_edgeRefresher_forceUpdate()
        else:
            self.def_edgeRefresher_noForceUpdate()
            
    def edgeInitialize(self, gr, ndataInputModule, args=None):
        return self.edgeRefresher.createEdge(gr, args)

    # f and g should be updated in user-defined class
    def f(self, t, y, gr, ndataInputModule, args=None):
        gr = self.edgeRefresher(gr, y, ndataInputModule, ndataOutputModule, args)
        return self.calc_module.f(t, gr, ndataInputModule, args)

    def g(self, t, y, gr, ndataInputModule, args=None):
        gr = self.edgeRefresher(gr, y, ndataInputModule, ndataOutputModule, args)
        return self.calc_module.g(t, gr, ndataInputModule, args)
