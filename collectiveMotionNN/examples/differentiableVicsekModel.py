import torch
from torch import nn

import collections

import collectiveMotionNN.utils as ut
import collectiveMotionNN.graph_utils as gu

 
class interactionModule(nn.Module):
    def __init__(self, v0, w0, sigma=0.1, d=1, positionName=None, velocityName=None, polarityName=None, torqueName=None, noiseName=None, messageName=None):
        super().__init__()
        
        self.v0 = nn.Parameter(torch.tensor(v0, requires_grad=True))
        self.w0 = nn.Parameter(torch.tensor(w0, requires_grad=True))

        self.sigma = nn.Parameter(torch.tensor(sigma, requires_grad=True))
        
        self.d = d
        
        self.prepare_sigma()
                    
        self.positionName = ut.variableInitializer(positionName, 'x')
        self.velocityName = ut.variableInitializer(velocityName, 'v')
        self.polarityName = ut.variableInitializer(polarityName, 'theta')
        self.torqueName = ut.variableInitializer(torqueName, 'w')
        self.noiseName = ut.variableInitializer(noiseName, 'sigma')
        
        self.messageName = ut.variableInitializer(messageName, 'm')
        
    def reset_parameter(self, v0=None, w0=None, sigma=None):
        if v0 is None:
            nn.init.uniform_(self.v0)
        else:
            nn.init.constant_(self.v0, v0)
            
        if w0 is None:
            nn.init.uniform_(self.w0)
        else:
            nn.init.constant_(self.w0, w0)
            
        if sigma is None:
            nn.init.uniform_(self.sigma)
        else:
            nn.init.constant_(self.sigma, sigma)

        
        self.prepare_sigma()
        
    def prepare_sigma(self):
        self.sigmaMatrix = torch.cat((torch.zeros([2,1], device=self.sigma.device), self.sigma*torch.ones([1,1], device=self.sigma.device)), dim=0)
            
    def calc_message(self, edges):
        dtheta = (edges.src[self.polarityName] - edges.dst[self.polarityName]) * self.d
        return {self.messageName: torch.cat((torch.cos(dtheta), torch.sin(dtheta)), -1)}
    
    def aggregate_message(self, nodes):
        mean_cs = torch.mean(nodes.mailbox[self.messageName], 1)
        return {self.torqueName : self.w0 * nn.functional.normalize(mean_cs, dim=-1)[..., 1:2]}
        
    def f(self, t, g, args=None):
        g.ndata[self.velocityName] = self.v0 * torch.cat((torch.cos(g.ndata[self.polarityName]), torch.sin(g.ndata[self.polarityName])), -1)
        g.update_all(self.calc_message, self.aggregate_message)
        return g
      
    def g(self, t, g, args=None):
        self.prepare_sigma()
        g.ndata[self.noiseName] = self.sigmaMatrix.repeat(g.ndata[self.positionName].shape[0], 1, 1).to(g.device)
        return g
    
class interactionModule_nonParametric(interactionModule):
    def __init__(self, fNNshape=None, fBias=None, sigma=0.1, gNNshape=None, gBias=None, positionName=None, velocityName=None, polarityName=None, torqueName=None, noiseName=None, messageName=None):
        super().__init__(0, 0, sigma, positionName, velocityName, polarityName, torqueName, noiseName, messageName)
        
        self.fNNshape = ut.variableInitializer(fNNshape, [128, 128, 128])
        self.gNNshape = ut.variableInitializer(gNNshape, None)
        
        self.fBias = ut.variableInitializer(fBias, True)
        self.gBias = ut.variableInitializer(gBias, True)
        
        self.init_fg()
        
    def createNNsequence(self, N_in, NNshape, N_out, bias):
        NNseq = collections.OrderedDict([])
        for i, NN_inout in enumerate(zip([N_in]+NNshape, NNshape+[N_out])):
            NNseq['Linear'+str(i)] = nn.Linear(NN_inout[0], NN_inout[1], bias=bias)
            NNseq['ReLU'+str(i)] = nn.ReLU()
        NNseq.pop('ReLU'+str(i))
        
        return nn.Sequential(NNseq)
    
    def init_fg(self):
        self.fNN = self.createNNsequence(4, self.fNNshape, 3, self.fBias)
        if self.gNNshape is None:
            self.gNN = None
        else:
            self.gNN = self.createNNsequence(4, self.gNNshape, 3, self.gBias)
            
    
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, len=None, delayTruth=1):
        super().__init__()
        
        self.dataPath = dataPath
        
        self.delayTruth = delayTruth
                
        if len is None:
            self.initialize()
        else:
            self.len = len
        
    def __len__(self):
        return self.len
      
    def loadData(self):
        x = torch.load(self.dataPath)
        return x.shape, x
      
    def initialize(self):
        self.extractDataLength = 1 + self.delayTruth
        
        xshape, _ = self.loadData()
        N_t, N_batch, N_particles, N_dim = xshape
        
        self.N_t = N_t
        self.N_batch = N_batch
        self.N_particles = N_particles
        self.N_dim = N_dim
        
        self.t_max = self.N_t - self.extractDataLength + 1
        
        self.len = self.t_max * self.N_batch
    
    def calc_t_batch(self, index):
        return divmod(index, self.t_max)
    
    def calc_t_batch_subset(self, index, batchIndices_subset):
        batch_sub, t = divmod(index, self.t_max)
        return batchIndices_subset[batch_sub], t
    
    def from_t_batch(self, batch, t):
        _, x = self.loadData()
        
        gr = gu.make_disconnectedGraph(x[t, batch], gu.multiVariableNdataInOut(['x', 'theta'], [2, 1]))
        
        x_truth = x[t+self.delayTruth, batch]
        
        return gr, x_truth
    
    def __getitem__(self, index):
        batch, t = self.calc_t_batch(index)
        return self.from_t_batch(batch, t)
    
class batchedSubset(torch.utils.data.Subset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices of batch in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        batch, t = self.dataset.calc_t_batch_subset(idx, self.indices)
        return self.dataset.from_t_batch(batch, t)

    def __len__(self):
        return len(self.indices) * self.dataset.t_max
    
    
    
class cosLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 1 - torch.cos(x - y).mean()
    
class myLoss(nn.Module):
    def __init__(self, distanceCalc):
        super().__init__()
        
        self.distanceCalc = distanceCalc
                
        self.xyLoss = nn.MSELoss()
        self.thetaLoss = cosLoss()
        
    def forward(self, x, y, score_x, score_y):
        dxy = self.distanceCalc(x[..., :2], y[..., :2])
        xyLoss = self.xyLoss(dxy, torch.zeros_like(dxy))
        thetaLoss = self.thetaLoss(x[..., 2], y[..., 2])
        scoreLoss = torch.mean(torch.square(torch.sum(score_x, dim=-1, keepdim=True) - score_y))
        return xyLoss, thetaLoss, scoreLoss
