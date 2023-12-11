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
            gaussiankl = calculateKL(rep2[i],rep1[j])
            # if i == 11 and j == 36:
                # print('here')
                # print(topgene2[11],topgene1[36])
            similarityDE = getSimilarity(topgene2,topgene1,i,j)
            #print(gaussiankl,similarityDE)
            #print('second:',i,'first',j)
            distance[i].append([gaussiankl , similarityDE])
    for i in range(len(distance)):
        distance[i] = normalizeDistance(distance[i])
#         print(distance[i])
#         print('---')
#         print('i=',i)
#         print('min',min(distance[i]))
#         print('min idx', distance[i].tolist().index(min(distance[i])))
    return distance
def connectNodes(distances):
    edges = []
    for i in range(len(distances)):
        leftend = np.argmin(distances[i])
        temp = distances[i].copy()
        pval = norm.cdf(distances[i][leftend],loc = np.mean(distances),scale = np.std(distances))
        #p*count/(count-idx)
        # q_val = pval * len(temp)/
        
        if pval < 0.01: #if pval < 0.01 can connect the two clusters across two stages
            edges.append([leftend,i])
    return edges

def buildEdges(stage1,stage2):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance
    args: 
    stage1,
    stage2
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
    edges = connectNodes(distance)
    return edges
def buildEdges(stage1,stage2,midpath,iteration):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance with midpath in iterative training
    args: 
    stage1,
    stage2
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
    edges = connectNodes(distance)
    return edges
def getandUpadateEdges(total_stage,midpath,iteration):
    '''
    get edges with midpath in iterative training
    '''
    edges = []
    for i in range(total_stage-1):
        edges.append(buildEdges(i,i+1,midpath,iteration))
    updateEdges(edges,midpath,iteration)
    print('edges updated')
    return edges

def getEdges():
    '''
    get edges
    '''
    edges = []
    for i in range(3):
        edges.append(buildEdges(i,i+1))
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