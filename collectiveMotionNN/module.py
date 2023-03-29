from torch import nn

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu







class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None, dynamicName=None, derivativeName=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
        self.dynamicName = ut.variableInitializer(dynamicName, 'y')
            
        self.derivativeName = ut.variableInitializer(derivativeName, 'v')
        
    def loadGraph(self, dynamicVariable, staticVariables, dynamicName=None):
        self.graph = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName)

        self.dynamicName = ut.variableInitializer(dynamicName, self.dynamicName)
                
    def dynamicValues(self):
        return self.graph(self.dynamicName)

    def f(self, t, y):
        self.graph = self.dynamicGNDEmodule.f(t, y, self.graph, self.dynamicName, self.derivativeName)
        return self.graph.ndata[self.derivativeName]
    
    def forward(self, t, y):
        return self.f(t, y)


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, noise_type, sde_type, graph=None, dynamicName=None, noiseName=None):
        super().__init__(dynamicGNDEmodule, graph, dynamicName, derivativeName)
        
        self.noiseName = ut.variableInitializer(noiseName, 's')

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y):
        self.graph = self.dynamicGNDEmodule.g(t, y, self.graph, self.dynamicName, self.noiseName)
        return self.graph.ndata[self.noiseName]



    
    
    
    
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
            

    # f and g should be updated in user-defined class
    def f(self, t, y, gr, dynamicName, derivativeName):
        gr = self.edgeRefresher(gr, y, dynamicName)
        return self.calc_module.f(t, gr, dynamicName, derivativeName)

    def g(self, t, y, gr, dynamicName, noiseName):
        gr = self.edgeRefresher(gr, y, dynamicName)
        return self.calc_module.g(t, gr, dynamicName, noiseName)
