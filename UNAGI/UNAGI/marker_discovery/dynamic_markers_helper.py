'''
this script is used to shuffle the gene expression for each stage data and use the shuffled data to build the random background to calculate the p-val for dynamic markers
'''
import scanpy as sc
import numpy as np
from ..dynamic_regulatory_networks.processIDREM import getClusterPaths,getClusterIdrem
def averageNode(nodes,state):
    '''
    calculate the average gene expression of sibling nodes
    args: 
    nodes: number of sibling nodes
    state: the gene expression of each cluster in a certain stage
    
    return: 
    out: the average gene expression of sibling nodes
    '''
    out = 0
    for each in nodes:
        out+=state[each]
    return out/len(nodes)
def get_progressionmarker_background(times, adata,total_stage):
    '''
    sampling and simulate the random background for dynamic markers.
    args:
    times: the number of times to sample the random background
    adata: the single-cell data
    total_stage: the total number of time stages

    return:
    results: the simulated random background
    '''
    results = {}
    edges = adata.uns['edges']
    paths = getClusterPaths(edges,total_stage)
    
    stage_data =[]
    for i in range(len(adata.obs['stage'].unique())):
        adata.obs['stage'] = adata.obs['stage'].astype(str)
        temp = adata[adata.obs[adata.obs['stage']==str(i)].index.tolist()]
        stage_data.append(temp)
    for time in range(times):
        print('shuffled times:',time)
        # results.append([])
        #shuffle the gene expression for each stage data and each gene
        avg_expression_val = []
        for i in range(len(stage_data)):
            avg_expression_val.append([])
            tempX = stage_data[i].X.toarray()
          
            # for j in range(stage_data[i].shape[1]):
            #     np.random.shuffle(tempX[:,j])
            np.random.shuffle(tempX)
            stage_data[i].layers['shuffle_X'] = tempX
            for cluster in range(len(stage_data[i].obs['leiden'].unique())):
                avg_expression_val[i].append(np.mean(stage_data[i][stage_data[i].obs[stage_data[i].obs['leiden']==str(cluster)].index.tolist()].layers['shuffle_X'],axis=0))
            del stage_data[i].layers['shuffle_X']

        idrem = np.array(getClusterIdrem(paths,avg_expression_val,total_stage))
        path = [each for each in paths.values() if len(each) == total_stage]
        for i, each in enumerate(path):
            track_name = ''
            for j, stage in enumerate(each):
                if j > 0:
                    track_name += '-'
                for k, cluster in enumerate(stage):
                    if k > 0:
                        track_name += 'n'
                    track_name += str(cluster)
            if track_name not in results.keys():
                results[track_name] = []
            results[track_name].append(idrem[i][:,-1] - idrem[i][:,0])
    return results
# out = permutation(permutation_time,adata)
# out = np.array(out).reshape(-1,adata.X.shape[1])
# print(out.shape)
# np.save('progressionMarker_background.npy',out)