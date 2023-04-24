import torch
from torch import nn

import inspect

import codecs


def variableInitializer(val, defaultVal):
    if val is None:
        return defaultVal
    else:
        return val

    
class euclidDistance_nonPeriodic(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, r_target, r_zero):
        return r_target - r_zero
    
class euclidDistance_periodic(nn.Module):
    def __init__(self, periodicLength):
        super().__init__()
        
        if torch.is_tensor(periodicLength):
            self.periodicLength = periodicLength
        else:
            self.periodicLength = torch.tensor(periodicLength, dtype=torch.float32)
        
    def forward(self, r_target, r_zero):
        dr = torch.remainder(r_target - r_zero, self.periodicLength)
        return dr - (dr > self.periodicLength/2) * self.periodicLength
    
    
def extractBrownian(bm, tW, device=None):
    bm_size = list(bm.size())
    Nt = len(tW) - 1
    device = variableInitializer(device, 'cpu')

    B_t = torch.zeros([Nt+1]+bm_size)
    for i in range(Nt):
        B_t[i+1] = B_t[i] + bm(tW[i], tW[i+1]).to(device).detach()

    return B_t


def getArgs():
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args}


def dict2txt(savePath, savedDict):

    txtstring = []
    for key in savedDict.keys():
        txtstring.append("{}, {}".format(key, savedDict[key]))

    print(*txtstring, sep="\n", file=codecs.open(savePath, 'w', 'utf-8'))

