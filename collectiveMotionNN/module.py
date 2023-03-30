from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu







class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInOutModule=None, velocityInOutModule=None, args=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
        self.ndataInOutModule = ut.variableInitializer(ndataInOutModule, gu.singleVariableNdataInOut('y'))
            
        self.velocityInOutModule = ut.variableInitializer(velocityInOutModule, gu.singleVariableNdataInOut('v'))

        self.edgeInitialize(args)
        
    def loadGraph(self, dynamicVariable, staticVariables):
        self.graph = gu.make_disconnectedGraph(dynamicVariable, staticVariables, self.ndataInOutModule)
    
    def edgeInitialize(self, args=None):
        self.graph = self.dynamicGNDEmodule.edgeInitialize(self.graph, self.ndataInOutModule, args)

    def f(self, t, y, args=None):
        self.graph = self.dynamicGNDEmodule.f(t, y, self.graph, self.ndataInOutModule, args)
        return self.ndataInOutModule.output(self.graph)
    
    def forward(self, t, y, args=None):
        return self.f(t, y, args)


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, graph=None, ndataInOutModule=None, velocityInOutModule=None, noiseInOutModule=None, args=None):
        super().__init__(dynamicGNDEmodule, graph, ndataInOutModule, velocityInOutModule, args)
        
        self.noiseInOutModule = ut.variableInitializer(noiseInOutModule, gu.singleVariableNdataInOut('s'))

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y, args=None):
        self.graph = self.dynamicGNDEmodule.g(t, y, self.graph, self.ndataInOutModule, args)
        return self.noiseInOutModule.output(self.graph)



    
    
    
    
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
            
    def edgeInitialize(self, gr, args=None):
        return self.edgeRefresher.createEdge(gr, args)

    def f(self, t, y, gr, ndataInOutModule, args=None):
        gr = self.edgeRefresher(gr, y, ndataInOutModule, args)
        return self.calc_module.f(t, gr, ndataInOutModule, args)

    def g(self, t, y, gr, ndataInOutModule, args=None):
        gr = self.edgeRefresher(gr, y, ndataInOutModule, args)
        return self.calc_module.g(t, gr, ndataInOutModule, args)
