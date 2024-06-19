import scanpy as sc
import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import rankdata
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import anndata
import os
from ..dynamic_graphs.distDistance import getClusterRepresentation

def get_data_file_path(filename):
    '''
    get the path of data file

    parameters
    ------------------
    filename: str
        name of the file

    return
    ------------------
    file_path: str
        path of the file
    '''
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', filename)
    return file_path

def split_dataset_into_stage(adata_path, folder, key):
    '''
    split dataset into stages and write to the path
    
    parameters
    ------------------
    adata: IPF database
    key: key of the dataset
    path: path to write the dataset

    '''
    adata = sc.read_h5ad(adata_path)
    for each in list(adata.obs[key].unqiue()):
        adata_temp = adata[adata.obs[key] == each]
        adata_temp.write_h5ad(os.path.join(folder,'%s.h5ad'%each),compression='gzip',compression_opts=9)
def transfer_to_ranking_score(gw):
    '''
    transfer gene weight to ranking score

    parameters
    ------------------
    gw: np.array
        gene weight of each gene in each cell

    return
    ------------------
    score: np.array
        ranking score of each gene in each cell
    '''

    od = gw.shape[1]-rankdata(gw,axis=1)+1
    score = 1+1/np.power(od,0.5)
    
    return score

def clustertype40(adata):
    '''
    annotate the cluster with cells >40% if no one >40%, annotate with the highest one
    
    parameters
    ------------------
    adata: anndata of one cluster
    
    return
    ---------------- 
    anootate:
        The cluster type
    '''
    dic = {}
    total = 0
    
    for each in adata.obs['name.simple']:
        if each not in dic.keys():
            dic[each]=1
        else:
            dic[each]+=1
        total+=1
    #print(dic)
   
    anootate = ''
    
    flag = False #flag to see if there are more than 1 cell types > 40%
    for each in list(dic.keys()):
        
        if dic[each] > total*0.5:
            if flag == False:
                anootate+=each
                flag = True
            else:
                anootate+='/'+each
    if flag == False:
        for each in list(dic.keys()):
        
            if dic[each] > total*0.4:
                if flag == False:
                    anootate+=each
                    flag = True
                else:
                    anootate+='/'+each
        if flag == False:
            for each in list(dic.keys()):
        
                if dic[each] > total*0.3:
                    if flag == False:
                        anootate+=each
                        flag = True
                    else:
                        anootate+='/'+each
        if flag == False:
            for each in list(dic.keys()):
        
                if dic[each] > total*0.2:
                    if flag == False:
                        anootate+=each
                        flag = True
                    else:
                        anootate+='/'+each
        if flag == False:
            anootate = 'Mixed'
    
    return anootate

def changeCluster(adata, ids, newIds):
    '''
    re-assign cluster id to cells

    parameters
    ------------------
    adata: anndata
        IPF database
    ids: int
        cell id in one stage needed to be changed 
    newIds: int
        new cluster id of cells

    return
    ------------------
    adata: anndata
        the updated IPF data
    '''

    for i, cluster in enumerate(ids):
        for j, each in enumerate(cluster):
            #print('vamos[i][j]',vamos[i][j])
            adata.obs['leiden'][each] = newIds[i][j]
    return adata
def extracth5adcluster(data, ID):
    '''
    extract data from a certain cluster

    parameters
    ------------------
    data: anndata
        data (H5AD class) of a certain stage
    ID: int
        the id of cluster
    
    return
    ------------------
    splitData: anndata
        data of a certain cluster
    splitDataID: list
        index of cells in a certain cluster
    '''
    splitDataID = []
    splitData = []
    for i in range(len(data)):
        
        if int(data.obs['leiden'][i]) == ID:
            splitDataID.append(i)
    splitData = data[splitDataID]
    
    return splitData,splitDataID

def extracth5adclusterids(data, ID):
    '''
    extract index from a certain cluster
    parameters
    ------------------
    data: anndata
        data (H5AD class) of a certain stage
    ID: int
        the id of cluster
    
    return:
    splitDataID: list
        index of cells in a certain cluster
    '''
    splitDataID = []
    splitData = []
    for i in range(len(data)):
        
        if int(data.obs['leiden'][i]) == ID:
            splitDataID+=data[i].obs.index.tolist()
    splitData = data[splitDataID]
    
    return splitDataID

def updateAttributes(adata,reps):
    '''
    update stage, cluster id of each stage, top 100 differential genes, cell types of clusters to anndata

    parameters
    ------------------
    adata: anndata
        anndata of database
    reps: list
        representations of sampled gaussian data points
    
    
    return
    ------------------ 
    adata: anndata
        updated anndata of database
    rep: list
        representations of sampled gaussian data points
    average_cluster: np.array
        average expression of each cluster
    '''
    print('top gene')
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon',n_genes=100)
    print('done')
    clusters = set(adata.obs['leiden'])
    average_cluster= []
    TDG = []
    rep = []
    celltypes = []
    
    # adata.obs['name.simple'] = adata.obs['name.simple'].astype('string')
    for i in range(len(clusters)):
        
        clusteradata,ids = extracth5adcluster(adata,i)
        clutertype = clustertype40(clusteradata) #use >40 strategy
        celltypes.append(clutertype)
        one_mean = np.mean(clusteradata.X,axis =0)
        one_mean = np.reshape(np.array(one_mean),-1)
        average_cluster.append(one_mean)
        adata.obs['ident'] = adata.obs['ident'].astype(str)
        TDG.append(list(adata.uns['rank_genes_groups']['names'][str(i)]))

        rep.append(getClusterRepresentation(reps[0][ids],reps[1][ids]))
        ids = extracth5adclusterids(adata,i)
        adata.obs.loc[adata.obs.index.isin(ids), "ident"] = clutertype
        # print('cluster type showing')
        # print(clutertype)
        # print(adata[ids].obs['ident'])
    adata.obs['ident'] = adata.obs['ident'].astype(str)
    adata.uns['clusterType'] = celltypes
    adata.uns['topGene'] = TDG
    return adata,average_cluster,rep

def saveRep(rep,midpath,iteration):
    '''
    write latent representations 'Z' of all stages to disk in iterative training

    parameters
    ------------------
    rep: list
        representations of sampled gaussian data points
    midpath: str
        directory to the task
    iteration: int
        iteration number
    '''
    np.save(os.path.join(midpath,str(iteration)+'/stagedata/rep.npy'),np.array(rep,dtype=object))

def get_all_adj_adata(adatas):
    '''
    merge all stages to a whole one

    parameters
    ------------------
    adatas: list
        list of anndata of each stage

    return
    ------------------
    adata: anndata
        anndata of the whole dataset
    '''
    data_size = []
    obs = []
    for i, each in enumerate(adatas):
        each.obs['stage'] = i
        obs.append(each.obs)
        each.obsp['gcn_connectivities'] = each.obsp['gcn_connectivities'].tocoo()
        if i == 0:
            if sp.isspmatrix(each.X):
                X = each.X
            else:
                X = csr_matrix(each.X)
            data_size.append(each.obsp['gcn_connectivities'].shape[0])
            gcn_data = each.obsp['gcn_connectivities'].data.tolist()
            col = each.obsp['gcn_connectivities'].col.tolist()
            row = each.obsp['gcn_connectivities'].row.tolist()
            if 'geneWeight' in each.layers.keys():
                geneWeight = each.layers['geneWeight']
            if 'X_pca' in each.obsm.keys():
                pca = each.obsm['X_pca']

        else:
            X = sp.vstack((X, each.X), format='csr')
            
            gcn_data += each.obsp['gcn_connectivities'].data.tolist()
            col += (data_size[i-1] + each.obsp['gcn_connectivities'].col).tolist()
            row += (data_size[i-1] + each.obsp['gcn_connectivities'].row).tolist()
            data_size.append(each.obsp['gcn_connectivities'].shape[0] + data_size[i - 1])
            if 'geneWeight' in each.layers.keys():
                geneWeight = sp.vstack((geneWeight, each.layers['geneWeight']), format='csr')
            if 'X_pca' in each.obsm.keys():
                pca = np.concatenate((pca,each.obsm['X_pca']),axis=0)
        each.obsp['gcn_connectivities'] = each.obsp['gcn_connectivities'].tocsr()
    
    obs = pd.concat(obs)
    variable = adatas[0].var 
    adata = anndata.AnnData(X=X,obs=obs,var=variable)
    gcn = csr_matrix((gcn_data, (row,col)), shape=(adata.X.shape[0],adata.X.shape[0]))

    if 'geneWeight' in each.layers.keys():
        adata.layers['geneWeight'] = geneWeight
    adata.obsp['gcn_connectivities'] = gcn
    if 'X_pca' in each.obsm.keys():
        adata.obsm['X_pca'] = pca
  
    return adata

def mergeAdata(path,total_stages):
    '''
    merge all stages to a whole one dataset and write to disk

    parameters
    ------------------
    path: str
        directory to the task
    total_stages: int
        total number of stages
    '''
    #read stage datasets
    adatas = []
    for i in range(total_stages):
        adata = sc.read_h5ad(os.path.join(path, 'stagedata/%d.h5ad'%i))
        adata.obs['stage'] = i
        adatas.append(adata)
    for i, each in enumerate(adatas):
        adatas[i].obsp['gcn_connectivities'] = adatas[i].obsp['gcn_connectivities'].tocoo()

    
    gcn_data = adatas[0].obsp['gcn_connectivities'].data.tolist()
    col = adatas[0].obsp['gcn_connectivities'].col.tolist()
    row = adatas[0].obsp['gcn_connectivities'].row.tolist()
    for i in range(1,total_stages):

        current_shape = sum([adatas[i].obsp['gcn_connectivities'].shape[0] for i in range(i)])
        col+=(current_shape+ adatas[i].obsp['gcn_connectivities'].col).tolist()
        row+=(current_shape+ adatas[i].obsp['gcn_connectivities'].row).tolist()
        gcn_data+=adatas[i].obsp['gcn_connectivities'].data.tolist()

    #get X
    if sp.isspmatrix(adatas[0].X):
        X = adatas[0].X
    else:
        X = csr_matrix(adatas[0].X)
    for i in range(1,total_stages):
        X = sp.vstack((X, adatas[i].X), format='csr')
    
    #get geneWeight
    geneWeight = adatas[0].layers['geneWeight']
    for i in range(1,total_stages):
        geneWeight = sp.vstack((geneWeight, adatas[i].layers['geneWeight']), format='csr')

    #get var
    if 'concat' in adatas[0].layers.keys():
        concat = adatas[0].layers['concat']
        for i in range(1,total_stages):
            concat = sp.vstack((concat, adatas[i].layers['concat']), format='csr')

    
    variable = adatas[0].var 
    
    #get obs
    obs = [each.obs for each in adatas]
    
    obs = pd.concat(obs)

    #get adata.ump['z']
    Z = adatas[0].obsm['z']
    for i in range(1,total_stages):
        Z = np.concatenate((Z,adatas[i].obsm['z']),axis=0)
    #get adata.ump['umap']
    umap = adatas[0].obsm['X_umap']
    for i in range(1,total_stages):
        umap = np.concatenate((umap,adatas[i].obsm['X_umap']),axis=0)
    #get top Gene
    topGene = {}
    top_gene_fold_change = {}
    top_gene_pvals_adj = {}
    clustertype = {}
    for i in range(total_stages):
        topGene[str(i)] = adatas[i].uns['topGene']
        clustertype[str(i)]=adatas[i].uns['clusterType']
    #top gene fold_change 
        adatas[i].uns['logfoldchanges'] = []
        adatas[i].uns['top_gene_pvals_adj'] = []
        for j in set(adatas[i].obs['leiden']):
            adatas[i].uns['logfoldchanges'].append(adatas[i].uns['rank_genes_groups']['logfoldchanges'][str(j)])
            adatas[i].uns['top_gene_pvals_adj'].append(adatas[i].uns['rank_genes_groups']['pvals_adj'][str(j)])

    #get uns.edges
    edges = eval(open(os.path.join(path,'edges.txt')).read())
    clusterType = clustertype
    #build new anndata and assign attribtues and write dataset
    adata = anndata.AnnData(X=X,obs=obs,var=variable)
    adata.layers['geneWeight'] = csr_matrix(geneWeight)
    adata.uns['clusterType']=clusterType
    adata.uns['edges']=edges
    adata.uns['topGene']=topGene
    adata.uns['top_gene_fold_change']=top_gene_fold_change
    adata.uns['top_gene_pvals_adj']=top_gene_pvals_adj
    adata.obsm['z']=Z
    adata.obsm['umap']=umap
    adata.obsm['X_umap'] = umap
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)
    adata.obsp = {}

    gcn = csr_matrix((gcn_data, (row,col)), shape=(adata.X.shape[0],adata.X.shape[0]))
    adata.obsp['gcn_connectivities'] = gcn 
    attribute = adata.uns
    with open(os.path.join(path,'stagedata/org_attribute.pkl'),'wb') as f:
        pickle.dump(attribute, f)
    del adata.uns
    adata.write_h5ad(os.path.join(path,'stagedata/org_dataset.h5ad'),compression='gzip',compression_opts=9)
