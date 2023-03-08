from torch import nn

import graph_utils as gu


class dynamicGODEwrapper(nn.Module):
    def __init__(self, module_f):
        super().__init__()

        self.module_f = module_f

    def loadGraph(self, dynamicVariable, staticVariables, dynamicName='y', batchID=None):
        self.graph, self.dynamicName, self.isBatched = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName, batchIDName=batchIDName)

    def f(self, t, y):
        return self.module_f.f(t, y, self.graph, self.dynamicName, self.isBatched)


class dynamicGSDEwrapper(nn.Module):
    def __init__(self, module_fg, noise_type, sde_type):
        super().__init__()

        self.module_fg = module_fg

        self.noise_type = noise_type
        self.sde_type = sde_type

    def loadGraph(self, dynamicVariable, staticVariables, dynamicName='y', batchID=None):
        self.graph, self.dynamicName, self.isBatched = gu.make_disconnectedGraph(dynamicVariable, staticVariables, dynamicName=dynamicName, batchID=batchID)

    def f(self, t, y):
        return self.module_fg.f(t, y, self.graph, self.dynamicName, self.isBatched)

    def g(self, t, y):
        return self.module_fg.g(t, y, self.graph, self.dynamicName, self.isBatched)



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
