import dgl
import dgl.function as fn



def make_heterograph(data_dict, future_edge_types=[]):
    '''
    data_dict: dictionary. Keys are the string indicating edge type, 
                values are tuples including two tensors indicating source and destination nodes.
    future_edge_types (optional): list. Edge types to be used in future should be included.
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

