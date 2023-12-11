import scanpy as sc
import numpy as np
from sklearn.neighbors import kneighbors_graph
def get_neighbors(stagedata,num_cells,anchor_neighbors,max_neighbors,min_neighbors):
    '''
    get neighbor parameters for each satge
    '''
    temp_num_cells = num_cells.copy()
    temp_num_cells.sort()
    median_index = int(len(temp_num_cells)/2)
    anchor = temp_num_cells[median_index]

    anchor_index = num_cells.index(anchor)
    temp_adata = stagedata[anchor_index]
    # distance = temp_adata.obsp['distances'].toarray()
    distance = temp_adata.obsp['distances'][temp_adata.obsp['distances'].nonzero()].reshape(len(temp_adata),-1)
    distance[:,::-1].sort()
    distance = distance[:,:max_neighbors]
    distance.sort()

    avg_anchor_distance = np.mean(distance[:,anchor_neighbors-1])
    neighbors = []
    for i in range(len(num_cells)):
        if i != anchor_index:
            temp_adata = stagedata[i]
            #distance = temp_adata.obsp['distances'].toarray()
            distance = temp_adata.obsp['distances'][temp_adata.obsp['distances'].nonzero()].reshape(len(temp_adata),-1)
            distance[:,::-1].sort()
            distance = distance[:,:max_neighbors]
            distance.sort()
            search = []
            for neighbor in range(max_neighbors):
                search.append(abs(np.mean(distance[:,neighbor])-avg_anchor_distance))
            #find the index of minimum value in search
            min_index = search.index(min(search))+1
            if min_index<min_neighbors:
                min_index=min_neighbors

            
            neighbors.append(min_index)
        else:
            neighbors.append(anchor_neighbors)
    return neighbors, anchor_index

def get_mean_median_cell_population(adata):
    num_cells = []
    for i in list(adata.obs['leiden'].unique()):
        temp = adata.obs[adata.obs['leiden'] == i].index.tolist()
        temp = adata[temp]
        num_cells.append(len(temp))
    return np.mean(num_cells)/len(adata), np.median(num_cells)/len(adata)
def auto_resolution(stagedata, anchor_index,neighbors, min_res, max_res):
    anchor_adata = stagedata[anchor_index]
    anchor_adata.obsp['connectivities'] = kneighbors_graph(anchor_adata.obsm['z'], neighbors[anchor_index], mode='connectivity', include_self=True,n_jobs=20)
    anchor_adata.obsp['distances'] = kneighbors_graph(anchor_adata.obsm['z'], neighbors[anchor_index], mode='distance', include_self=True,n_jobs=20)
    sc.tl.leiden(anchor_adata, resolution = min_res)
    all_means = []
    anchor_mean, anchor_median = get_mean_median_cell_population(anchor_adata)
    out_res = []
    for i in range(len(stagedata)):
        differences = []
        temp_all_means = []
        if i != anchor_index:
            temp_adata = stagedata[i]
            temp_adata.obsp['connectivities'] = kneighbors_graph(temp_adata.obsm['z'], neighbors[i], mode='connectivity', include_self=True,n_jobs=20)
            temp_adata.obsp['distances'] = kneighbors_graph(temp_adata.obsm['z'], neighbors[i], mode='distance', include_self=True,n_jobs=20)
            for j in np.arange(min_res,max_res+0.1,0.1):
                sc.tl.leiden(temp_adata, resolution = j)
                temp_mean, temp_median = get_mean_median_cell_population(temp_adata)
                temp_all_means.append(temp_mean)
                differences.append(abs(temp_mean-anchor_mean))
            min_index = differences.index(min(differences))
            out_res.append(min_index*0.1+min_res)
            all_means.append(temp_all_means[min_index])
        else:
            out_res.append(1)
            all_means.append(anchor_mean)
    return out_res, all_means