#this script is used to shuffle the gene expression for each stage data 
#and and use the shuffled data to build the random background to calculate the p-val for progression markers
import scanpy as sc
import numpy as np
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
def getClusterPaths(edges):
    '''
    obtain the paths of each clusters
    args:
    edeges: contains three lists of edges between control group and IPF 1 stage, IPF 1 stage and IPF 2 stage, IPF 2 stage and IPF 3 stage
    
    return:
    paths: a collection of paths of clusters
    '''
    paths = {}
    C2oneEdge = edges['0']
    one2twoEdge = edges['1']
    two2threeEdge = edges['2']
    for each in C2oneEdge:
        if str(each[0]) not in paths.keys():
            paths[str(each[0])]=[[each[0]],[each[1]]]
        else:
            paths[str(each[0])][1].append(each[1])

    #connect2 = {}
    for each in one2twoEdge:
        for item in paths.keys():
            if each[0] in paths[item][1]:
                if len(paths[item]) == 2:
                    paths[item].append([each[1]])
                else:
                    paths[item][2].append(each[1])
                

    for each in two2threeEdge:
        for item in paths.keys():
            if len(paths[item]) == 2:
                continue
            if each[0] in paths[item][2]:
                if len(paths[item]) == 3:
                    paths[item].append([each[1]])
                else:
                    paths[item][3].append(each[1])  
    return paths
def getClusterIdrem(paths,state):
    '''
    concatenate the average gene expression in a cluster tree. shape: [number of stages, number of gene]
    args: 
    paths: the collection of paths
    state: a list of average gene expression of each state
    
    return: 
    out: a list of gene expression of each cluster tree
    '''
    out = []
    for i,each in enumerate(paths.keys()):
        if len(paths[each]) == 4:
            stages = []
            for j, item in enumerate(paths[each]):
                
                stages.append(averageNode(item,state[j]))
            joint_matrix = np.concatenate((stages[0].reshape(-1,1), stages[1].reshape(-1,1), stages[2].reshape(-1,1), stages[3].reshape(-1,1)),axis =1)
            out.append(joint_matrix)
    return out
# data_dir = 'iterativeTrainingNOV26/4/stagedata/kkdataset.h5ad'
# permutation_time = 1000
# adata = sc.read(data_dir)
def get_progressionmarker_background(times, adata):
    results = {}
    edges = adata.uns['edges']
    paths = getClusterPaths(edges)
    
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

        idrem = np.array(getClusterIdrem(paths,avg_expression_val))
        path = [each for each in paths.values() if len(each) == 4]
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