import torch
from torch import nn
from torch.utils.data import Dataset

import dgl
import dgl.function as fn


class graphDataLoader(fun, radius_edge, periodic=False, periodicLength=None, addSelfLoop=False, MinkowskiMetric=2):
    '''
    This DataLoader makes radius graph from dataset when loading. This feature suppresses memory usage.
    '''
    def __init__(self):
        self.periodic = periodic
        self.periodicLength = periodicLength
        
        self.addSelfLoop = addSelfLoop
        self.radius_edge = radius_edge
        self.MinkowskiMetric = MinkowskiMetric
        
    def calc_dr_periodic(self, r1, r2):
        dr = (r1 - r2) % self.periodicLength
        dr[dr > self.periodicLength/2] = dr[dr > self.periodicLength/2] - self.periodicLength
        return dr
    
    def make_graph(self, x):
        if self.periodic:
            Ndata = x.size(0)
            dr = self.calc_dr_periodic(x, x.T)
            drsq = np.linalg.norm(dr, ord=self.MinkowskiMetric, axis=-1, keepdims=False)
            if self.addSelfLoop:
                edges = np.argwhere(drsq < self.radius_edge)
            else:
                edges = np.argwhere(np.logical_and(drsq > 0, drsq < self.radius_edge))
            return dgl.graph((edges[:,0], edges[:,1]), num_nodes=Ndata)
        else:
            return dgl.radius_graph(x, self.radius_edge, p=self.MinkowskiMetric, self_loop=self.addSelfLoop)
      
    

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
