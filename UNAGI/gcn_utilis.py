import numpy as np
from sklearn.neighbors import kneighbors_graph 
import scanpy as sc
import torch
import os
def find_neighbourhood(adj, start, end):
    '''
    find and return the neighbourhoods of cells in the batch)
    '''
    # adj = adj.to_dense()
    adj = adj.numpy()

    targets = [i for i in range(start, end)]
    out = targets.copy()
    for each in targets:
        
        out+=list(np.nonzero(adj[each])[0])
    out = list(set(out))
    return out
def new_find_neighbourhood(adj):
    '''
    find and return the neighbourhoods of cells in the batch)
    '''
    # adj = adj.to_dense()
    # adj = adj.numpy()
    out = []
    for each in adj:
        temp = each.indices.tolist()
        out.append(temp)
    return out


def setup_graph(adj):
    '''
    transfer adj into coo and set up self weight
    '''
    # adj.setdiag(1)
    # adj = adj.asformat('coo')
    adj.data = adj.data
#     adj.setdiag(1)
    adj_values = adj.data
    adj_indices = np.vstack((adj.row, adj.col))
    adj_shape = adj.shape
    adj_i = torch.LongTensor(adj_indices)
    adj_v = torch.FloatTensor(adj_values)
    adj = torch.sparse.FloatTensor(adj_i, adj_v, torch.Size(adj_shape))
    return adj
def get_gcn_exp(source_directory,total_stage,neighbors,threads= 20):
    '''
    get the gcn connectivities for each cell
    save stage adata with gcn connectivities in the same directory
    '''

    for i in range(total_stage):
        print('Calculating cell graph for stage %d.....'%i)
        read_path = source_directory+'/%d.h5ad'%i
        temp = sc.read_h5ad(read_path)
        
        if 'gcn_connectivities' in temp.obsp.keys():
            continue

        sc.pp.pca(temp)
        adj = kneighbors_graph(temp.obsm['X_pca'],  neighbors-1, mode='connectivity', include_self=False,n_jobs=threads)
        adj.setdiag(neighbors)#for pcls
        temp.obsp['gcn_connectivities'] = adj
        write_path = os.path.join(source_directory,'%d.h5ad'%i)
        temp.write(write_path, compression='gzip', compression_opts=9)

def get_neighbours(batch_size, adj, cell_loader):
    '''
    return a list of neighbors to speed up training
    '''
    neighbourhoods = []
    for i, x in enumerate(cell_loader):
        # temp_x = placeholder.clone()
        start = i*batch_size
        if (1+i)*batch_size > len(adj):
            end =  len(adj)
        else:
            end = (1+i)*batch_size
        neighbourhoods.append(find_neighbourhood(adj.to_dense(), start,end))
    return neighbourhoods