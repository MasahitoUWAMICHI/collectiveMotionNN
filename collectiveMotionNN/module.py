from torch import nn

import graph_utils as gu



class edgeRefresh_noForceUpdate(nn.Module):
    def __init__(self, edgeCondtionModule):
        super().__init__()
        
        self.edgeConditionModule = edgeConditionModule
        
    def forward(self, gr, dynamicVariable, dynamicName):
        if gu.judge_skipUpdate(gr, dynamicVariable, dynamicName):
            return gr
        else:
            return gu.edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule)

class edgeRefresh_forceUpdate(nn.Module):
    def __init__(self, edgeCondtionModule):
        super().__init__()
        
        self.edgeConditionModule = edgeConditionModule
        
    def forward(self, gr, dynamicVariable, dynamicName):
        return gu.edgeRefresh_execute(gr, dynamicVariable, dynamicName, self.edgeConditionModule)







class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None, dynamicName=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
        if dynamicName is None:
            self.dynamicName = 'y'
        else:
            self.dynamicName = dynamicName
        
    def loadGraph(self, dynamicVariable, staticVariables, dynamicName=None):
        self.graph = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName)
        if dynamicName is None:
            return None
        else:
            self.dynamicName = dynamicName
            return None

    def f(self, t, y):
        out, self.graph = self.dynamicGNDEmodule.f(t, y, self.graph, self.dynamicName)
        return out


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, noise_type, sde_type, graph=None, dynamicName=None):
        super().__init__(dynamicGNDEmodule, graph)

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y):
        out, self.graph = self.dynamicGNDEmodule.g(t, y, self.graph, self.dynamicName)
        return out



    
    
    
    
class dynamicGNDEmodule(nn.Module):
    def __init__(self, calc_module, edgeConditionModule, forceUpdate=False):
        super().__init__()
        
        self.calc_module = calc_module

        self.edgeConditionModule = edgeConditionModule
        
        self.forceUpdate = forceUpdate

        self.def_edgeRefresher()
        
    def def_edgeRefresher_forceUpdate(self):
        self.edgeRefresher = edgeRefresh_forceUpdate(self.edgeConditionModule)

    def def_edgeRefresher_noForceUpdate(self):
        self.edgeRefresher = edgeRefresh_noForceUpdate(self.edgeConditionModule)
        
    def def_edgeRefresher(self):
        if self.forceUpdate:
            self.def_edgeRefresher_forceUpdate()
        else:
            self.def_edgeRefresher_noForceUpdate()
            

    # f and g should be updated in user-defined class
    def f(self, t, y, gr, dynamicName):
        gr = self.edgeRefresher(gr, y, dynamicName)
        return self.calc_module.f(t, gr, dynamicName), gr

    def g(self, t, y, gr, dynamicName):
        gr = self.edgeRefresher(gr, y, dynamicName)
        return self.calc_module.g(t, gr, dynamicName), gr
