import scanpy as sc
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,davies_bouldin_score
import scib
from sklearn.neighbors import kneighbors_graph
import numpy as np

def neighbor_score(adata, embeddingA,type_label):
    embedding_A = [i.indices for i in embeddingA]
    neighbor_label = adata.obs.iloc[np.concatenate(embedding_A, axis=0)][type_label]
    cell_label = adata.obs.iloc[np.repeat(np.arange(len(embedding_A)), [len(j) for j in embedding_A])][type_label]
    n_shared_labels = (neighbor_label.values==cell_label.values).sum() / len(embedding_A)

    return  n_shared_labels
def get_label_score(adata,embedding,type_label):
    neighbors = kneighbors_graph(embedding, 50, mode='distance', include_self=True,n_jobs=25)
    label_score = neighbor_score(adata, neighbors,type_label)
    return label_score/50
def run_metrics_subsample(adatas, cell_type_key, stage_key,portion,random_state=0):
    '''
    Evaluation metrics.

    Parameters
    ----------
    adatas : AnnData object
        Annotated data matrix.
    cell_type_key : str
        Key for cell type column in adata.obs.
    stage_key : str
        Key for stage column in adata.obs.

    Returns
    --------------

    '''
    print('using portion: ', portion)
    consistency = []
    ariss= []
    NMIs = []
    silhouettes = []
    # ITERATION= 5
    stage_keys = adatas.obs[stage_key].unique().tolist()
    stage_keys = sorted(stage_keys)
    stage_keys = stage_keys[::-1]
    total_stage = len(stage_keys)
    total_adata = 0
    count=0
    NMI = 0
    DBI = 0
    label_scores = 0
    silhouettes =0
    aris = 0
    isolated_asws = 0
    cell_type_asws = 0
    isolated_labels_f1s = 0
    for i,stage in enumerate(stage_keys):
        
        temp_count = 0
        #check the type of adatas.obs[stage_key]
        if adatas.obs[stage_key].dtype == 'str':
            stage = str(stage)
        elif adatas.obs[stage_key].dtype == 'int':
            stage = int(stage)
        print(len(adatas.obs[adatas.obs[stage_key] == stage].index.tolist()))

        adata = adatas[adatas.obs[adatas.obs[stage_key] == stage].index.tolist()]
        sc.pp.subsample(adata, fraction=portion, copy=False,random_state=random_state)
        adata.obs['UNAGI'] = adata.obs[cell_type_key].astype('category')
        adata.obs['leiden'] = adata.obs['leiden'].astype('category')
        adata.obs['leiden'] = adata.obs['leiden'].astype('string')
        total_adata+=len(adata)
        count+=temp_count
        ari = adjusted_rand_score(adata.obs['name.simple'],adata.obs['UNAGI'] )
        nmi =  normalized_mutual_info_score(adata.obs['name.simple'],adata.obs['UNAGI'])
        dbi = davies_bouldin_score(adata.obsm['z'],adata.obs['leiden'])
        label_score = get_label_score(adata,adata.obsm['z'],'name.simple')
        silhouette = silhouette_score(adata.obsm['z'], adata.obs['leiden'])
        isolated_asw = scib.metrics.isolated_labels_asw(adata,label_key = 'name.simple', batch_key = 'stage', embed='z')
        celltype_asw = silhouette_score(adata.obsm['z'], adata.obs['name.simple'])
        isolated_f1 = scib.metrics.isolated_labels_f1(adata,label_key = 'name.simple', batch_key = 'stage', embed='z')
        print('ARI: ', ari)
        print('NMIs: ', nmi)
        print('DBI:',dbi)
        print('label_score: ', label_score)
        print('silhouette score: ', silhouette)
        print('isolated_asw: ', isolated_asw)
        print('isolated_f1: ', isolated_f1)
        print('celltype_asw: ', celltype_asw)
        isolated_asws += isolated_asw
        isolated_labels_f1s += isolated_f1
        NMI += nmi
        DBI += dbi
        label_scores+=label_score
        silhouettes += silhouette
        aris += ari
        cell_type_asws += celltype_asw


    print('ARIs: ', aris/total_stage)
    print('NMI: ', NMI/total_stage)
    print('DBIs:', DBI/total_stage)
    print('label_scores: ', label_scores/total_stage)
    print('silhouette score: ', silhouettes/total_stage)
    print('isolated_asw: ', isolated_asws/total_stage)
    print('celltype_asw: ', cell_type_asws/total_stage)
    print('isolated_f1: ', isolated_labels_f1s/total_stage)
    return aris/total_stage, NMI/total_stage,DBI/total_stage,label_scores/total_stage, silhouettes/total_stage, isolated_asws/total_stage, cell_type_asws/total_stage, isolated_labels_f1s/total_stage
def run_metrics(adatas, cell_type_key, stage_key):
    '''
    Evaluation metrics.

    Parameters
    ----------
    adatas : AnnData object
        Annotated data matrix.
    cell_type_key : str
        Key for cell type column in adata.obs.
    stage_key : str
        Key for stage column in adata.obs.

    Returns
    --------------

    '''
    consistency = []
    ariss= []
    NMIs = []
    silhouettes = []
    # ITERATION= 5
    stage_keys = adatas.obs[stage_key].unique().tolist()
    stage_keys = sorted(stage_keys)
    stage_keys = stage_keys[::-1]
    total_stage = len(stage_keys)
    total_adata = 0
    count=0
    NMI = 0
    DBI = 0
    label_scores = 0
    silhouettes =0
    aris = 0
    isolated_asws = 0
    cell_type_asws = 0
    isolated_labels_f1s = 0
    for i,stage in enumerate(stage_keys):
        
        temp_count = 0
        #check the type of adatas.obs[stage_key]
        if adatas.obs[stage_key].dtype == 'str':
            stage = str(stage)
        elif adatas.obs[stage_key].dtype == 'int':
            stage = int(stage)
        print(len(adatas.obs[adatas.obs[stage_key] == stage].index.tolist()))

        adata = adatas[adatas.obs[adatas.obs[stage_key] == stage].index.tolist()]
        adata.obs['UNAGI'] = adata.obs[cell_type_key].astype('category')
        adata.obs['leiden'] = adata.obs['leiden'].astype('category')
        adata.obs['leiden'] = adata.obs['leiden'].astype('string')
        total_adata+=len(adata)
        count+=temp_count
        ari = adjusted_rand_score(adata.obs['name.simple'],adata.obs['UNAGI'] )
        nmi =  normalized_mutual_info_score(adata.obs['name.simple'],adata.obs['UNAGI'])
        dbi = davies_bouldin_score(adata.obsm['z'],adata.obs['leiden'])
        label_score = get_label_score(adata,adata.obsm['z'],'name.simple')
        silhouette = silhouette_score(adata.obsm['z'], adata.obs['leiden'])
        isolated_asw = scib.metrics.isolated_labels_asw(adata,label_key = 'name.simple', batch_key = 'stage', embed='z')
        celltype_asw = silhouette_score(adata.obsm['z'], adata.obs['name.simple'])
        isolated_f1 = scib.metrics.isolated_labels_f1(adata,label_key = 'name.simple', batch_key = 'stage', embed='z')
        print('ARI: ', ari)
        print('NMIs: ', nmi)
        print('DBI:',dbi)
        print('label_score: ', label_score)
        print('silhouette score: ', silhouette)
        print('isolated_asw: ', isolated_asw)
        print('isolated_f1: ', isolated_f1)
        print('celltype_asw: ', celltype_asw)
        isolated_asws += isolated_asw
        isolated_labels_f1s += isolated_f1
        NMI += nmi
        DBI += dbi
        label_scores+=label_score
        silhouettes += silhouette
        aris += ari
        cell_type_asws += celltype_asw


    print('ARIs: ', aris/total_stage)
    print('NMI: ', NMI/total_stage)
    print('DBIs:', DBI/total_stage)
    print('label_scores: ', label_scores/total_stage)
    print('silhouette score: ', silhouettes/total_stage)
    print('isolated_asw: ', isolated_asws/total_stage)
    print('celltype_asw: ', cell_type_asws/total_stage)
    print('isolated_f1: ', isolated_labels_f1s/total_stage)
    return aris/total_stage, NMI/total_stage,DBI/total_stage,label_scores/total_stage, silhouettes/total_stage, isolated_asws/total_stage, cell_type_asws/total_stage, isolated_labels_f1s/total_stage