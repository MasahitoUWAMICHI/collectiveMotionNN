from torch import nn

import graph_utils as gu


class dynamicGODEwrapper(nn.Module):
    def __init__(self, dynamicGNDEmodule, graph=None):
        super().__init__()

        self.dynamicGNDEmodule = dynamicGNDEmodule

        self.graph = graph
        
    def loadGraph(self, dynamicVariable, staticVariables, dynamicName='y'):
        self.graph, self.dynamicName = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName)

    def f(self, t, y):
        return self.dynamicGNDEmodule.f(t, y, self.graph, self.dynamicName)


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, dynamicGNDEmodule, noise_type, sde_type, graph=None):
        super().__init__(dynamicGNDEmodule, graph)

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y):
        return self.dynamicGNDEmodule.g(t, y, self.graph, self.dynamicName)



class dynamicGNDEmodule(nn.Module):
    def __init__(self, calc_module, edgeConditionFunc, forceUpdate=False):
        super().__init__()
        
        self.calc_module = calc_module

        self.edgeConditionFunc = edgeConditionFunc
        
        self.forceUpdate = forceUpdate


    def edgeRefresh(self, gr, dynamicVariable, dynamicName):
        if self.forceUpdate:
            gr = self.edgeRefresh_execute(gr, dynamicVariable, dynamicName)
            return gr
        else:
            if gu.judge_skipUpdate(gr, dynamicVariable, dynamicName):
                return gr
            else:
                gr = self.edgeRefresh_execute(gr, dynamicVariable, dynamicName)
                return gr

    # f and g should be updated in user-defined class
    def f(self, t, y, gr, dynamicName):
        gr = self.edgeRefresh(gr, dynamicVariable, dynamicName)
        return self.calc_module.f(t, y, gr, dynamicName)

    def g(self, t, y, gr, dynamicName):
        gr = self.edgeRefresh(gr, dynamicVariable, dynamicName)
        return self.calc_module.g(t, y, gr, dynamicName)
