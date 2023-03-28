from torch import nn

import graph_utils as gu


class dynamicGODEwrapper(nn.Module):
    def __init__(self, module_f, graph=None):
        super().__init__()

        self.module_f = module_f

        self.graph = graph
        
    def loadGraph(self, dynamicVariable, staticVariables, dynamicName='y'):
        self.graph, self.dynamicName = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName)

    def f(self, t, y):
        return self.module_f.f(t, y, self.graph, self.dynamicName, self.isBatched)


class dynamicGSDEwrapper(dynamicGODEwrapper):
    def __init__(self, module_f, noise_type, sde_type, graph=None):
        super().__init__(module_f, graph)

        self.noise_type = noise_type
        self.sde_type = sde_type
        
    def g(self, t, y):
        return self.module_f.g(t, y, self.graph, self.dynamicName, self.isBatched)



class dynamicGNDEmodule(nn.Module):
    def __init__(self, edgeConditionFunc):
        super().__init__()

        self.edgeConditionFunc = edgeConditionFunc

    def edgeRefresh_execute(self, gr, dynamicVariable, dynamicName):
        gr.ndata[dynamicName] = dynamicVariable
        gu.update_adjacency(gr, self.edgeConditionFunc)

    def edgeRefresh(self, gr, dynamicVariable, dynamicName, forceUpdate=False):
        if forceUpdate:
            self.edgeRefresh_execute(gr, dynamicVariable, dynamicName)
        else:
            if not(gu.judge_skipUpdate(gr, dynamicVariable, dynamicName)):
                self.edgeRefresh_execute(gr, dynamicVariable, dynamicName)

    # f and g should be updated in user-defined class
    def f(self, t, y, gr, dynamicName, isBatched):
        return None

    def g(self, t, y, gr, dynamicName, isBatched):
        return None
