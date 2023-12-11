import scanpy as sc
import numpy as np
import umap
import pandas as pd
import os
import pickle
from scipy.stats import rankdata
import torch
from .distDistance import getClusterRepresentation
from sklearn.neighbors import kneighbors_graph
def get_data_file_path(filename):
    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)
    return file_path
def split_dataset_into_stage(adata_path, folder, key):
    '''
    split dataset into stages and write to the path
    args:
    adata: IPF database
    key: key of the dataset
    path: path to write the dataset
    
    return:
    None
    '''
    adata = sc.read_h5ad(adata_path)
    for each in list(adata.obs[key].unqiue()):
        adata_temp = adata[adata.obs[key] == each]
        adata_temp.write_h5ad(os.path.join(folder,'%s.h5ad'%each),compression='gzip',compression_opts=9)
def transfer_to_ranking_score(gw):
    '''
    ranking score
    '''
    # gw = adata.layers['geneWeight'].toarray()
    od = gw.shape[1]-rankdata(gw,axis=1)+1
    score = 1+1/np.power(od,0.5)
    
    return score
def clustertype_old(adata):
    '''
    find the most common cell types to represent the cluster
    args:
    adata: anndata of one cluster
    
    return: the most common cell types in the cluster
    '''
    dic = {}
    for each in adata.obs['ident']:
        if each not in dic.keys():
            dic[each]=1
        else:
            dic[each]+=1
    #print(dic)        
    return max(dic, key=dic.get)
def clustertype40(adata):
    '''
    annotate the cluster with cells >40% if no one >40%, annotate with the highest one
    args:
    adata: anndata of one cluster
    
    return: the most common cell types in the cluster
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

def getInTrackNode(idrem_path,stage):
    filenames = os.listdir(idrem_path)
    nodes = [[] for _ in range(4)]
    for each in filenames:
        temp = each.split('.')[0].split('-')
        for i,item in enumerate(temp):
            temp1 = item.split('n')
            nodes[i].append(temp1)
    return nodes[stage]
def getInconsistency(adata):
    T=0
    for i in range(4):
    
        adata = sc.read_h5ad('./stagedata/%d-test.h5ad'%i)
        adata.obs['ident'] = 'None'
        for i,each in enumerate(adata.obs['name.simple']):
            each = each.split('_')
            adata.obs['ident'][i] = each[0]

        adata.obs['clustertype']='None'
        sc.pp.neighbors(adata, use_rep="z",n_neighbors=15,method='rapids')
        sc.tl.leiden(adata,resolution=0.7)
        reducer=umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        zr=reducer.fit_transform(adata.obsm['z'])
        adata.obsm["umap"]=zr
    
    
        #sc.pl.umap(adata,color="lineage.ident")
        adata.obs['clustertype']=adata.obs['clustertype'].astype('string')
        for each in set(adata.obs['leiden']):
        
            clusteradataid = adata.obs[adata.obs['leiden']==each].index.tolist()
            clusteradata=adata[clusteradataid]
            celltype = clustertype(clusteradata)
            #print(celltype)
            adata.obs['clustertype'][clusteradataid] = celltype
        sc.pl.umap(adata,color="ident")
        sc.pl.umap(adata,color="clustertype")
        T+=calculateinconsistency(adata)
    #print('celltype inconsistency: ',calculateinconsistency(adata))
    print(T)
def changeCluster(adata, ids, newIds):
    '''
    re-assign cluster id to cells
    args: 
    adata: IPF database
    ids: cell id in one stage needed to be changed 
    vamos: new cluster id of cells
    return: 
    out: new IPF database
    '''

    for i, cluster in enumerate(ids):
        for j, each in enumerate(cluster):
            #print('vamos[i][j]',vamos[i][j])
            adata.obs['leiden'][each] = newIds[i][j]
    return adata
def extracth5adcluster(data, ID):
    '''
    extract data from a certain cluster
    args:
    data: data (H5AD class) of a certain stage
    ID: the id of cluster
    
    return:
    splitData: data of a certain cluster
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
    args:
    data: data (H5AD class) of a certain stage
    ID: the id of cluster
    
    return:
    splitData: data of a certain cluster
    '''
    splitDataID = []
    splitData = []
    for i in range(len(data)):
        
        if int(data.obs['leiden'][i]) == ID:
            splitDataID+=data[i].obs.index.tolist()
    splitData = data[splitDataID]
    
    return splitDataID

def retrieveResults(adata, ids,stage):
    '''
    retrieve results from adata
    args:
    adata: IPF database
    ids: id of cells in each stage
    stage: stage id
    out:
    Rep
    cluster_type
    top_gene
    average_value
    '''
    Rep = []
    average_value = []
    cluster_type = []
    stageids = adata.obs[adata.obs['stage'] == stage].index.tolist()
    stageadata = adata[stageids]
    stageadata = sc.tl.rank_genes_groups(stageadata, 'leiden', method='wilcoxon',n_genes=100,copy=True)

    top_gene = []
    idskeys = list(ids.keys())
    clusteridset = set(stageadata.obs['leiden'].values)


    for i in ids.keys():
        if int(i) not in clusteridset:
            print('miss1')
            print(i)
            #ids.pop(i, None)
            continue
        tempid = stageadata.obs[stageadata.obs['leiden'] == int(i)].index.tolist()
        if len(tempid) == 0:
            #ids.pop(i, None)
            print('miss2')
            print(i)
            continue
        cluster = stageadata[tempid]
        Rep.append(getClusterRepresentation(cluster.obs['mu'].values,cluster.obs['sigma'].values,cluster.obs['theta'].values))
        average_value.append(np.mean(cluster.X,axis =0))
        cluster_type.append(clustertype(cluster))
        top_gene.append(list(stageadata.uns['rank_genes_groups']['names'][str(i)]))
    print('retrive ok')
    return Rep, cluster_type, top_gene, average_value
def getAvgClusgerRep(mid, iteration):
    avg = []
    for i in range(4):
        tempavg = []
        adata = sc.read_h5ad('./'+mid+'/'+str(iteration)+'/stagedata/%d.h5ad'%(i))
        for clusterid in set(adata.obs['leiden']):
            clusteradataid = adata.obs[adata.obs['leiden'] == clusterid].index.tolist()
            clusteradata = adata[clusteradataid]
            clusteravg = np.mean(clusteradata.X, axis = 0)
            clusteravg = np.reshape(np.array(clusteravg),-1)
            tempavg.append(clusteravg)
        tempavg = np.array(tempavg)
        avg.append(tempavg)
    avg = np.array(avg)
    return avg
def updateAttributes(adata,reps):
    '''
    update stage, cluster id of each stage, top 100 differential genes, cell types of clusters to anndata

    args: 
    adata: anndata of database
    stage: the IPF stage of this fold of data
    
    
    returns: 
    adata: updated anndata of database
    rep: representations of sampled gaussian data points
    average_cluster: cluster.X average values
    '''
    #stageids = adata.obs[adata.obs['stage'] == stage].index.tolist()
    #stageadata = adata[stageids]
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
def saveRep(rep):
    '''
    write rep of all stages to disk
    '''
    np.save('./stagedata/rep.npy',np.array(rep,dtype=object))
def saveRep(rep,midpath,iteration):
    '''
    write rep of all stages to disk with midpath in iterative training
    '''
    np.save(os.path.join(midpath,str(iteration)+'/stagedata/rep.npy'),np.array(rep,dtype=object))

import scipy.sparse as sp
from scipy.sparse import csr_matrix
import anndata
import os
def getUnsDict(adata0,adata1,adata2,adata3,key):
    dic = {}
    dic['0'] = adata0.uns[key]
    dic['1'] = adata1.uns[key]
    dic['2'] = adata2.uns[key]
    dic['3'] = adata3.uns[key]
    return dic
def mergeReconstructionAdata(path):
    '''
    merge adata file from different stage to a whole one
    '''
    #read stage datasets
    # adata0 = sc.read_h5ad('./'+path+'/stagedata/0.h5ad')
    # adata1 = sc.read_h5ad('./'+path+'/stagedata/1.h5ad')
    # adata2 = sc.read_h5ad('./'+path+'/stagedata/2.h5ad')
    # adata3 = sc.read_h5ad('./'+path+'/stagedata/3.h5ad')
    adata0 = sc.read_h5ad('./may21_MAY14_8_reconstruction_0.h5ad')
    adata1 = sc.read_h5ad('./may21_MAY14_8_reconstruction_1.h5ad')
    adata2 = sc.read_h5ad('./may21_MAY14_8_reconstruction_2.h5ad')
    adata3 = sc.read_h5ad('./may21_MAY14_8_reconstruction_3.h5ad')
    #get X
    X = adata0.X
    X = sp.vstack((X, adata1.X), format='csr')
    X = sp.vstack((X, adata2.X), format='csr')
    X = sp.vstack((X, adata3.X), format='csr')
    
    #get geneWeight
    if 'geneWeight'  in adata0.layers.keys():
        
        geneWeight = adata0.layers['geneWeight']
        geneWeight = sp.vstack((geneWeight, adata1.layers['geneWeight']), format='csr')
        geneWeight = sp.vstack((geneWeight, adata2.layers['geneWeight']), format='csr')
        geneWeight = sp.vstack((geneWeight, adata3.layers['geneWeight']), format='csr')
    else:
        geneWeight= None
    #get var
    variable = adata1.var

    #set adata.obs['stage']    
    adata0.obs['stage'] = 0
    adata1.obs['stage'] = 1
    adata2.obs['stage'] = 2
    adata3.obs['stage'] = 3
    
    #get obs
    obs = [adata0.obs,adata1.obs,adata2.obs,adata3.obs]
    
    obs = pd.concat(obs)

    #get adata.ump['z']
    if 'z' in adata0.obsm.keys():
        Z = np.concatenate((adata0.obsm['z'],adata1.obsm['z']),axis=0)
        Z = np.concatenate((Z,adata2.obsm['z']),axis=0)
        Z = np.concatenate((Z,adata3.obsm['z']),axis=0)
    #get adata.ump['umap']
    else:
        Z= None
    hcmarkers = getUnsDict(adata0,adata1,adata2,adata3,'hcmarkers')
    linkage_matrix = getUnsDict(adata0,adata1,adata2,adata3,'linkage_matrix')
    linkage_order = getUnsDict(adata0,adata1,adata2,adata3,'linkage_order')
    annotation_meta = getUnsDict(adata0,adata1,adata2,adata3,'annotation_meta')
    
    umap = np.concatenate((adata0.obsm['X_umap'],adata1.obsm['X_umap']),axis=0)
    umap = np.concatenate((umap,adata2.obsm['X_umap']),axis=0)
    umap = np.concatenate((umap,adata3.obsm['X_umap']),axis=0)
    #get top Gene
    
    topGene = {}
    topGene['0']=adata0.uns['topGene']
    topGene['1']=adata1.uns['topGene']
    topGene['2']=adata2.uns['topGene']
    topGene['3']=adata3.uns['topGene']
    #top gene fold_change 
    adata0.uns['logfoldchanges'] = []
    for i in set(adata0.obs['leiden']):
        adata0.uns['logfoldchanges'].append(adata0.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata1.uns['logfoldchanges'] = []
    for i in set(adata1.obs['leiden']):
        adata1.uns['logfoldchanges'].append(adata1.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata2.uns['logfoldchanges'] = []
    for i in set(adata2.obs['leiden']):
        adata2.uns['logfoldchanges'].append(adata2.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata3.uns['logfoldchanges'] = []
    for i in set(adata3.obs['leiden']):
        adata3.uns['logfoldchanges'].append(adata3.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    top_gene_fold_change = {}
    top_gene_fold_change['0']=adata0.uns['logfoldchanges']
    top_gene_fold_change['1']=adata1.uns['logfoldchanges']
    top_gene_fold_change['2']=adata2.uns['logfoldchanges']
    top_gene_fold_change['3']=adata3.uns['logfoldchanges']
    #get uns.edges
    edges = eval(open('./'+path+'/edges.txt').read())
    #get clusterType

    clustertype = {}
    
    clustertype['0']=adata0.uns['clusterType']
    clustertype['1']=adata1.uns['clusterType']
    clustertype['2']=adata2.uns['clusterType']
    clustertype['3']=adata3.uns['clusterType']
    clusterType = clustertype
    #build new anndata and assign attribtues and write dataset
    adata = anndata.AnnData(X=X,obs=obs,var=variable)
    #adata.layers['geneWeight'] = csr_matrix(geneWeight)
    adata.uns['clusterType']=clusterType
    adata.uns['edges']=edges
    adata.uns['topGene']=topGene
    adata.obsm['z']=Z
    adata.obsm['umap']=umap
    adata.obsm['X_umap'] = umap
    adata.obs['leiden'] = adata.obs['leiden'].astype('string')
    adata.uns['hcmarkers'] = hcmarkers
    adata.uns['linkage_matrix'] = linkage_matrix
    adata.uns['linkage_order'] = linkage_order
    adata.uns['annotation_meta'] = annotation_meta
    
    adata.write_h5ad('./may21_MAY14_8_reconstruction.h5ad',compression='gzip',compression_opts=9) 
def get_all_adj_adata(adatas):
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

    # adata0.obsp['gcn_connectivities'] = adata0.obsp['gcn_connectivities'].tocoo()
    # adata1.obsp['gcn_connectivities'] = adata1.obsp['gcn_connectivities'].tocoo()
    # adata2.obsp['gcn_connectivities'] = adata2.obsp['gcn_connectivities'].tocoo()
    # adata3.obsp['gcn_connectivities'] = adata3.obsp['gcn_connectivities'].tocoo()
    # gcn_data = adata0.obsp['gcn_connectivities'].data.tolist()
    # col = adata0.obsp['gcn_connectivities'].col.tolist()
    # row = adata0.obsp['gcn_connectivities'].row.tolist()
    # col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].col).tolist()
    # row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].row).tolist()
    # gcn_data+=adata1.obsp['gcn_connectivities'].data.tolist()
    # col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].col).tolist()
    # row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].row).tolist()
    # gcn_data+=adata2.obsp['gcn_connectivities'].data.tolist()
    # col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].shape[0]+ adata3.obsp['gcn_connectivities'].col).tolist()
    # row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].shape[0]+ adata3.obsp['gcn_connectivities'].row).tolist()
    # print(max(row))
    # gcn_data+=adata3.obsp['gcn_connectivities'].data.tolist()
def mergeAdata(path):
    '''
    merge adata file from different stage to a whole one
    '''
    #read stage datasets
    
    adata0 = sc.read_h5ad(os.path.join(path, 'stagedata/0.h5ad'))
    adata1 = sc.read_h5ad(os.path.join(path, 'stagedata/1.h5ad'))
    adata2 = sc.read_h5ad(os.path.join(path,'stagedata/2.h5ad'))
    adata3 = sc.read_h5ad(os.path.join(path,'stagedata/3.h5ad'))
    adata0.obsp['gcn_connectivities'] = adata0.obsp['gcn_connectivities'].tocoo()
    adata1.obsp['gcn_connectivities'] = adata1.obsp['gcn_connectivities'].tocoo()
    adata2.obsp['gcn_connectivities'] = adata2.obsp['gcn_connectivities'].tocoo()
    adata3.obsp['gcn_connectivities'] = adata3.obsp['gcn_connectivities'].tocoo()
    gcn_data = adata0.obsp['gcn_connectivities'].data.tolist()
    col = adata0.obsp['gcn_connectivities'].col.tolist()
    row = adata0.obsp['gcn_connectivities'].row.tolist()
    col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].col).tolist()
    row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].row).tolist()
    gcn_data+=adata1.obsp['gcn_connectivities'].data.tolist()
    col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].col).tolist()
    row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].row).tolist()
    gcn_data+=adata2.obsp['gcn_connectivities'].data.tolist()
    col+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].shape[0]+ adata3.obsp['gcn_connectivities'].col).tolist()
    row+=(adata0.obsp['gcn_connectivities'].shape[0]+ adata1.obsp['gcn_connectivities'].shape[0]+ adata2.obsp['gcn_connectivities'].shape[0]+ adata3.obsp['gcn_connectivities'].row).tolist()
    print(max(row))
    gcn_data+=adata3.obsp['gcn_connectivities'].data.tolist()

    #get X
    if sp.isspmatrix(adata0.X):
        X = adata0.X
    else:
        X = csr_matrix(adata0.X)
    X = sp.vstack((X, adata1.X), format='csr')
    X = sp.vstack((X, adata2.X), format='csr')
    X = sp.vstack((X, adata3.X), format='csr')
    
    #get geneWeight
    geneWeight = adata0.layers['geneWeight']
    geneWeight = sp.vstack((geneWeight, adata1.layers['geneWeight']), format='csr')
    geneWeight = sp.vstack((geneWeight, adata2.layers['geneWeight']), format='csr')
    geneWeight = sp.vstack((geneWeight, adata3.layers['geneWeight']), format='csr')
    #get var
    if 'concat' in adata0.layers.keys():
        concat = adata0.layers['concat']
        concat = sp.vstack((concat, adata1.layers['concat']), format='csr')
        concat = sp.vstack((concat, adata2.layers['concat']), format='csr')
        concat = sp.vstack((concat, adata3.layers['concat']), format='csr')
    
    variable = adata1.var 
    adata0.obs['stage'] = 0
    adata1.obs['stage'] = 1
    adata2.obs['stage'] = 2
    adata3.obs['stage'] = 3
    
    #get obs
    obs = [adata0.obs,adata1.obs,adata2.obs,adata3.obs]
    
    obs = pd.concat(obs)

    #get adata.ump['z']
    Z = np.concatenate((adata0.obsm['z'],adata1.obsm['z']),axis=0)
    Z = np.concatenate((Z,adata2.obsm['z']),axis=0)
    Z = np.concatenate((Z,adata3.obsm['z']),axis=0)
    #get adata.ump['umap']
    
    umap = np.concatenate((adata0.obsm['X_umap'],adata1.obsm['X_umap']),axis=0)
    umap = np.concatenate((umap,adata2.obsm['X_umap']),axis=0)
    umap = np.concatenate((umap,adata3.obsm['X_umap']),axis=0)
    #get top Gene
    topGene = {}
    topGene['0']=adata0.uns['topGene']
    topGene['1']=adata1.uns['topGene']
    topGene['2']=adata2.uns['topGene']
    topGene['3']=adata3.uns['topGene']
    #top gene fold_change 
    adata0.uns['logfoldchanges'] = []
    adata0.uns['top_gene_pvals_adj'] = []
    for i in set(adata0.obs['leiden']):
        adata0.uns['top_gene_pvals_adj'].append(adata0.uns['rank_genes_groups']['pvals_adj'][str(i)])
        adata0.uns['logfoldchanges'].append(adata0.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata1.uns['logfoldchanges'] = []
    adata1.uns['top_gene_pvals_adj'] = []
    for i in set(adata1.obs['leiden']):
        adata1.uns['top_gene_pvals_adj'].append(adata1.uns['rank_genes_groups']['pvals_adj'][str(i)])
        adata1.uns['logfoldchanges'].append(adata1.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata2.uns['logfoldchanges'] = []
    adata2.uns['top_gene_pvals_adj'] = []
    for i in set(adata2.obs['leiden']):
        adata2.uns['top_gene_pvals_adj'].append(adata2.uns['rank_genes_groups']['pvals_adj'][str(i)])
        adata2.uns['logfoldchanges'].append(adata2.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    adata3.uns['logfoldchanges'] = []
    adata3.uns['top_gene_pvals_adj'] = []
    for i in set(adata3.obs['leiden']):
        adata3.uns['top_gene_pvals_adj'].append(adata3.uns['rank_genes_groups']['pvals_adj'][str(i)])
        adata3.uns['logfoldchanges'].append(adata3.uns['rank_genes_groups']['logfoldchanges'][str(i)])
    top_gene_fold_change = {}
    top_gene_fold_change['0']=adata0.uns['logfoldchanges']
    top_gene_fold_change['1']=adata1.uns['logfoldchanges']
    top_gene_fold_change['2']=adata2.uns['logfoldchanges']
    top_gene_fold_change['3']=adata3.uns['logfoldchanges']
    #top gene pvals_adj
    top_gene_pvals_adj = {}
    top_gene_pvals_adj['0']=adata0.uns['top_gene_pvals_adj']
    top_gene_pvals_adj['1']=adata1.uns['top_gene_pvals_adj']
    top_gene_pvals_adj['2']=adata2.uns['top_gene_pvals_adj']
    top_gene_pvals_adj['3']=adata3.uns['top_gene_pvals_adj']
    #get uns.edges
    edges = eval(open(os.path.join(path,'edges.txt')).read())
    #get clusterType

    clustertype = {}
    
    clustertype['0']=adata0.uns['clusterType']
    clustertype['1']=adata1.uns['clusterType']
    clustertype['2']=adata2.uns['clusterType']
    clustertype['3']=adata3.uns['clusterType']
    clusterType = clustertype
    #build new anndata and assign attribtues and write dataset
    adata = anndata.AnnData(X=X,obs=obs,var=variable)
    adata.layers['geneWeight'] = csr_matrix(geneWeight)
    # if 'concat' in adata0.layers.keys():
        # adata.layers['concat'] = csr_matrix(concat)
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
    adata.obsp['gcn_connectivities'] = gcn #np.zeros(shape=(adata.X.shape[0],adata.X.shape[0]),dtype=np.float32)
    attribute = adata.uns
    with open(os.path.join(path,'stagedata/org_attribute.pkl'),'wb') as f:
        pickle.dump(attribute, f)
    del adata.uns
    adata.write_h5ad(os.path.join(path,'stagedata/org_dataset.h5ad'),compression='gzip' )
def getUnsDict(adata0,adata1,adata2,adata3,key):
    dic = {}
    dic['0'] = adata0.uns[key]
    dic['1'] = adata1.uns[key]
    dic['2'] = adata2.uns[key]
    dic['3'] = adata3.uns[key]
    return dic