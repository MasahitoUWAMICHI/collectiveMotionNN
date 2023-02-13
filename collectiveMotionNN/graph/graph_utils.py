import dgl
import dgl.function as fn

def update_edges(g, edges):
    g.remove_edges(g, g.edge_ids(g.edges()[0], g.edges()[1]))
    g.add_edges(edges)
    return g

def make_RadiusGraph(x, r, flg_selfloop=False, flg_custom=False, func_custom_distance=None):
    '''
    x : list of torch.Tensor. The shape must be [N_nodes, N_dimensions].
    r : real number or torch.Tensor. Two nodes will be linked with edge if the distance is smaller than r. If torch.Tensor, this must be able to broadcast to [N_nodes, N_nodes, N_dimensions].
    flg_selfloop (optional) : boolean. If True, the graph will include selfloops.
    flg_custom (optional) : boolean. If True, custom function will be used to calculate distance. If False, euclidean distance will be used.
    func_custom_distance (optional) : callable. Function to calculate distance from x.
    This returns a DGL graph.
    '''
    if flg_custom:
        Ndata = x[0].size(0)
        dx = func_custom_distance(x)
        if flg_selfloop:
            edges = torch.argwhere(dx < r)
        else:
            edges = torch.argwhere(torch.logical_and(dx > 0, dx < r))
        out = dgl.graph((edges[:,0], edges[:,1]), num_nodes=Ndata)
    else:
        out = dgl.radius_graph(x[0], r, p=2, self_loop=flg_selfloop)
    return out

def update_RadiusGraph(g, xy_list, r, flg_selfloop=False, flg_custom=False, func_custom_distance=None):

    x = [g[xy_name] for xy_name in xy_list]

    newgraph = make_RadiusGraph(x, r, flg_selfloop=False, flg_custom=False, func_custom_distance=None)
    update_edges(g, newgraph.adj_sparse('coo'))
    
    return g

def make_heterograph(data_dict, future_edge_types=[]):
    '''
    data_dict: dictionary. Keys are the string indicating edge type, 
                values are tuples including two tensors indicating source and destination nodes.
    future_edge_types (optional): list of strings. Edge type to be used in future should be included.
    '''
    edge_types_not_found = [etype for etype in future_edge_types if not(etype in data_dict)]

    for etype in edge_types_not_found:
        data_dict[etype] = (torch.tensor([0]), torch.tensor([0]))

    hg = dgl.heterograph(data_dict)

    for etype in edge_types_not_found:
        hg.remove_edges(0, etype=etype)

    return hg


def add_graph2heterograph(heterograph, graph, etype):
    '''
    add edge of graph(dgl.graph) to heterograph(dgl.heterograph) as the edge type indicated as etype(string).
    etype should be one of the edge types of heterograph.
    '''
    edge_to_be_added = graph.adj_sparse('coo')
    heterograph.add_edges(edge_to_be_added[0], edge_to_be_added[1], etype=etype)
    return heterograph

