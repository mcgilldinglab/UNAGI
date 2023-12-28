import numpy as np
import gc
import anndata
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KernelDensity
from sklearn import cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, entropy, multivariate_normal, gamma
from scipy import stats 
import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import entropy 
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import threading
from scipy.stats import multivariate_normal
import gc
import os
from .distDistance import *
def nodesDistance(rep1,rep2,topgene1,topgene2):
    '''
    calculate the distance between two stages
    args: 
    rep1: the representation of clusters in stage 1
    rep2: the representation of clusters in stage 2
    topgene1: top 100 differential gene of clusters in stage 1
    topgene2: top 100 differential gene of clusters in stage 2
    return:
    distance: normalized distance of clusters between two stages
    '''
    distance = [[] for _ in range(len(rep2))]
    for i in range(len(rep2)):
        for j in range(len(rep1)):
            gaussiankl_1 = calculateKL(rep2[i],rep1[j])
            gaussiankl_2 = calculateKL(rep1[j],rep2[i])
            gaussiankl = (gaussiankl_1 + gaussiankl_2)/2
            similarityDE = getSimilarity(topgene2,topgene1,i,j)
            distance[i].append([gaussiankl , similarityDE])
    for i in range(len(distance)):
        distance[i] = normalizeDistance(distance[i])
    return distance
def connectNodes(distances,cutoff = 0.05):
    '''
    Connect the clusters in two stages with smallest distance and p-value < cut-off
    '''
    edges = []
    for i in range(len(distances)):
        leftend = np.argmin(distances[i])
        # temp = distances[i].copy()
        pval = norm.cdf(distances[i][leftend],loc = np.mean(distances),scale = np.std(distances))
        #p*count/(count-idx)
        # q_val = pval * len(temp)/
        
        if pval < cutoff: #if pval < 0.01 can connect the two clusters across two stages
            edges.append([leftend,i])
    return edges

def buildEdges(stage1,stage2,cutoff = 0.05):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance
    args: 
    stage1: the anndata of the first selected stage
    stage2: the anndata of the second selected stage
    cutoff: the cutoff of p-value
    return:
    edges: the edges between two stages
    '''
    adata1 = sc.read_h5ad('./stagedata/%d.h5ad'%stage1)
    adata2 = sc.read_h5ad('./stagedata/%d.h5ad'%stage2)
    reps = np.load('./stagedata/rep.npy',allow_pickle=True)
    
    rep1 = reps[stage1]
    rep2 = reps[stage2]
    topgene1 = adata1.uns['topGene']
    topgene2 = adata2.uns['topGene']
    distance = nodesDistance(rep1,rep2,topgene1,topgene2)
    edges = connectNodes(distance,cutoff)
    return edges
def buildEdges(stage1,stage2,midpath,iteration,cutoff = 0.05):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance with midpath in iterative training
    args: 
    midpath: the path of the midpath
    iteration: the iteration of the training
    stage1: the anndata of the first selected stage
    stage2: the anndata of the second selected stage
    cutoff: the cutoff of p-value

    return:
    edges: the edges between two stages
    '''
    # print(stage1,stage2)
    adata1 = sc.read_h5ad(os.path.join(midpath,str(iteration)+'/stagedata/%d.h5ad'%stage1))
    adata2 = sc.read_h5ad(os.path.join(midpath,str(iteration)+'/stagedata/%d.h5ad'%stage2))
    reps = np.load(os.path.join(midpath,str(iteration)+'/stagedata/rep.npy'),allow_pickle=True)
    
    rep1 = reps[stage1]
    rep2 = reps[stage2]
    topgene1 = adata1.uns['topGene']

    topgene2 = adata2.uns['topGene']
    distance = nodesDistance(rep1,rep2,topgene1,topgene2)
    edges = connectNodes(distance,cutoff)
    return edges
def getandUpadateEdges(total_stage,midpath,iteration):
    '''
    get edges with midpath in iterative training
    '''
    edges = []
    for i in range(total_stage-1):
        edges.append(buildEdges(i,i+1,midpath,iteration))
    updateEdges(edges,midpath,iteration)
    # print('edges updated')
    return edges

def updateEdges(edges,midpath,iteration):
    '''
    updata edges to the anndata database, calculate edges changes
    args:
    adata: anndata of database
    edges: edges from buildEdges()

    return: 
    adata: updated anndata of database
    '''
    newEdges = {}
    for i in range(len(edges)):
        newEdges[str(i)] = edges[i]
    #adata.uns['edges'] = newEdges
    f = open(os.path.join(midpath,str(iteration)+'/edges.txt'),'w')
    f.write(str(newEdges))
    f.close()

def reupdateAttributes(adata, stage, results):
    '''
    update gaussian and gamma rep, top 100 differential genes, cell types of clusters to anndata

    args: 
    adata: anndata of database
    results: [gaussian, gamma], a list contained top differential genes and cell types of clusters
    
    returns: 
    adata: updated anndata of database
    '''
    stageids = adata.obs[adata.obs['stage'] == stage].index.tolist()
    tempadata = adata[stageids]
    adata.uns['rep'][str(stage)] = results[0]
    adata.uns['topGene'][str(stage)] = results[1]
    adata.uns['clusterType'][str(stage)] = results[2]
    return adata
