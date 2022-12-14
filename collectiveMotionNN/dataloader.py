import torch
from torch import nn
from torch.utils.data import Dataset

import dgl
import dgl.function as fn
from dgl.dataloading import GraphDataLoader

class graphDataset(Dataset):
    '''
    This Dataset makes radius graph from dataset when loading. This feature suppresses memory usage.
    
    dataDictList: List of dictionalies.
    
    '''
    def __init__(self, dataDictList, radiusEdge, delayData=0, delayTruth=1, 
                 keyToCalcDistance='x', keyToStoreData=['x'], keyToStoreTruth=['x'], 
                 periodic=False, periodicLength=None, addSelfLoop=False, MinkowskiMetric=2):
        super().__init__()
        self.dataDictList = dataDictList
        
        self.periodic = periodic
        self.periodicLength = periodicLength
        
        self.addSelfLoop = addSelfLoop
        self.radiusEdge = radiusEdge
        self.MinkowskiMetric = MinkowskiMetric

        self.delayData = delayData
        self.delayTruth = delayTruth
        self.keyToCalcDistance = keyToCalcDistance
        self.keyToStoreData = keyToStoreData
        self.keyToStoreTruth = keyToStoreTruth
        
        self.set_delays()
        
    def set_delays(self):
        if type(self.delayData) is int:
            self.maxDelayData = self.delayData
        else:
            self.maxDelayData = max(self.delayData)

        if type(self.delayTruth) is int:
            self.maxDelayTruth = self.delayTruth
        else:
            self.maxDelayTruth = max(self.delayTruth)

        self.maxItersData = [[self.dataDictList[i][key_x].size(0) - self.maxDelayData for key_x in self.keyToStoreData]\
                             for i in len(self.dataDictList)]
        self.maxItersTruth = [[self.dataDictList[i][key_y].size(0) - self.maxDelayTruth for key_y in self.keyToStoreData]\
                              for i in len(self.dataDictList)]
        
    def calc_dr_periodic(self, r1, r2):
        dr = torch.remainder(r1 - r2, self.periodicLength)
        dr[dr > self.periodicLength/2] = dr[dr > self.periodicLength/2] - self.periodicLength
        return dr
    
    def make_graph(self, x):
        if self.periodic:
            Ndata = x.size(0)
            dr = self.calc_dr_periodic(x, x.T)
            drsq = torch.norm(dr, p=self.MinkowskiMetric, dim=-1, keepdim=False)
            if self.addSelfLoop:
                edges = torch.argwhere(drsq < self.radius_edge)
            else:
                edges = torch.argwhere(torch.logical_and(drsq > 0, drsq < self.radius_edge))
            return dgl.graph((edges[:,0], edges[:,1]), num_nodes=Ndata)
        else:
            return dgl.radius_graph(x, self.radius_edge, p=self.MinkowskiMetric, self_loop=self.addSelfLoop)
      
    def __len__(self):
        
        return (self.data_len - (self.t_yseq - 1)).sum()
    
    def __getitem__(self, index):
        id_List = np.argwhere(index<self.data_len_cumsum)[0,0]
        
        if id_List:
            id_tensor = index - self.data_len_cumsum[id_List-1]
        else:
            id_tensor = index
        
        return self.data_x[id_List][id_tensor], self.data_x[id_List][id_tensor:(id_tensor+self.t_yseq)], self.celltype_List[id_List]  
            
        
   
    
    
def graphDataSet(fun_data, vars_fun, indices):
    '''
    '''
    train_x = []
    valid_x = []
    test_x = []

    train_y = []
    valid_y = []
    test_y = []

    for i_dir, subdirName in enumerate(datadir_list):

        traj = np.load(subdirName+'/result.npz')

        xy_t = torch.tensor(traj['xy'])#[:-1,:,:])
        p_t = torch.unsqueeze(torch.tensor(traj['theta']), dim=2)#[:-1,:]), dim=2)

        if i_dir in train_inds:
            train_x.append(torch.concat((xy_t, p_t), -1))
            train_ct.append(torch.tensor(traj['celltype_label']).view(-1,1))

        if i_dir in valid_inds:
            valid_x.append(torch.concat((xy_t, p_t), -1))
            valid_ct.append(torch.tensor(traj['celltype_label']).view(-1,1))

        if i_dir in test_inds:
            test_x.append(torch.concat((xy_t, p_t), -1))
            test_ct.append(torch.tensor(traj['celltype_label']).view(-1,1))

    train_dataset = myDataset(train_x, train_ct, t_yseq=T_pred)

    valid_dataset = myDataset(valid_x, valid_ct, t_yseq=T_pred)

    test_dataset = myDataset(test_x, test_ct, t_yseq=T_pred)
