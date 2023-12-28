'''
This module contains functions to identify dynamic genes and accumulate gene weights.
'''
import numpy as np
import gc
import anndata
import os
import json
from scipy.stats import rankdata
import subprocess
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
from scipy.sparse import lil_matrix
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

def getLastStageIDs(json,total_stage):
    '''
    get the list of node ids in last stage 
    args:
    json: idrem meta
    total_stage: total number of stages

    return:
    out: a list of node ids in last stage
    '''
    out = []
    for each in json[0][1:]:
        if each['nodetime'] == 'stage'+str(total_stage-1):
            out.append(each['nodeID'])
    return out
def getidremPath(json,cid):
    '''
    given the last node id in idrem, get the whole paths
    
    args:
    json: idrem meta file
    cid: the last node id in idrem

    return:
    nodes: a list of node ids in the path
    '''
    nodes = []

    for each in reversed(json[0][1:]):

        if each['nodeID'] == cid:
            nodes.append(cid)
            if each['parent']!= -1:
                cid = each['parent']
            else:
                break
    nodes.reverse()
    return nodes
def getIdremPaths(json,total_stage):
    '''
    get all paths in idrem
    
    args:
    json: idrem meta files

    return:
    paths: a list of all paths
    '''
    
    stage3IDs = getLastStageIDs(json,total_stage) 
    paths = []
    for i in stage3IDs:
        paths.append(getidremPath(json, i))
    return paths
def getPosNegMaxChangesNodes(json,paths,total_stage):
    means = []
    for path in paths:
        pathmean = []
        for each in path:
            
            for node in json[0][1:]:
                if node['nodeID'] == each:
                    pathmean.append(node['nodeMean'])
        means.append(pathmean)
    means = np.array(means)
    indicator = total_stage-1
    while indicator > 0:
        means[:,indicator] = means[:,indicator]-means[:,indicator-1]
        indicator-=1
    poschangenodesid = np.argmax(means,axis=0)
    negchangenodesid = np.argmax(-means,axis=0)
    poschangenodes = []
    negchangenodes = []
    for i, each in enumerate(poschangenodesid):
        poschangenodes.append(paths[each][i])
    for i, each in enumerate(negchangenodesid):
        negchangenodes.append(paths[each][i])
    return poschangenodes, negchangenodes
def testChanges(path,filename):
    '''
    a test function
    '''
    tt = readIdremJson(path, filename)
    paths = getIdremPaths(tt,4)

    return getPosNegMaxChangesNodes(tt,paths)
def getTargetGenes(path,N):
    '''
    get top N genes of each path
    args:
    path: the file path of IDREM results
    
    return:
    out: a list of top N up or down regulators of each path
    '''
    out = []

    filenames = os.listdir(path)
    
    for each in filenames:
        if each[0] != '.':
            #out.append(getMaxMinPathGenes(path,each,N)) #get the genes from nodes that have highest and lowest nodemean
            out.append(getPosNegDynamicPathGenes(path,each,N)) #get the genes from nodes increase and descrease most between stages
    return out

def getPosNegDynamicPathGenes(path,filename,topN):
    '''
    get the genes from nodes that increase and decrease most between stages
    args:
    path: the file path of IDREM results
    filename: the file name of IDREM results
    topN: the number of genes to accumulate gene weights

    return:
    out: a list of top N up or down regulators of each path
    '''
    total_stage = len(filename.split('.')[0].split('-'))
    tt = readIdremJson(path, filename)
    paths = getIdremPaths(tt,total_stage)
    posdynamicids, negdynamicids = getPosNegMaxChangesNodes(tt,paths,total_stage-1)
    posdynamicgenes,posdynamicgeneids = getMaxOrMinNodesGenes(tt,posdynamicids)
    negdynamicgenes,negdynamicgeneids = getMaxOrMinNodesGenes(tt,negdynamicids)
    posnegdynamicgenes = [posdynamicgenes[i]+negdynamicgenes[i] for i in range(total_stage-1)]
    posnegdynamicgeneids = [posdynamicgeneids[i]+negdynamicgeneids[i] for i in range(total_stage-1)]
    out = getTopNTargetGenes(tt,posnegdynamicgenes,posnegdynamicgeneids,topN, total_stage-1)
    return out

def getMaxOrMinNodesGenes(json,nodes):
    '''
    get the genes from dynamic nodes
    args:
    json: idrem meta
    nodes: the list of dynamic nodes

    return:
    genes: a list of genes in the nodes
    '''
    boolgenes = []
    genes = []
    geneids = []

    for each in json[0]:
        if each['nodeID'] in nodes:
            boolgenes.append(each['genesInNode'])
    for each in boolgenes:
        tempgenes = []
        tempgeneids = []
        for i, gene in enumerate(each):
            if gene == True:
                tempgenes.append(json[3][i].upper())
                tempgeneids.append(i)
        genes.append(tempgenes)
        geneids.append(tempgeneids)
    return genes,geneids

# def getMaxOrMinPathGenes(json,nodes):
#     boolgenes = []
#     genes = []
#     geneids = []
#     for each in json[0][1:]:
#         if each['nodeID'] in nodes:
#             boolgenes.append(each['genesInNode'])
#     for each in boolgenes:

#         for i, gene in enumerate(each):
#             if gene == True:
#                 genes.append(json[3][i])
#                 geneids.append(i)
#     genes = list(set(genes))
#     geneids = list(set(geneids))
#     return genes,geneids
def readIdremJson(path, filename):
    '''
    Parse the IDREM json file
    args:
    path: the file path of IDREM results
    filename: the file name of IDREM results

    return:
    tt: the parsed IDREM json file
    '''
    print('getting Target genes from ', filename)
    path = os.path.join(path,filename,'DREM.json')
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    return tt

def getTopNTargetGenes(json,genenames,geneids,topN,total_stage):
    '''
    get top N genes of each path sorted by the change of gene expression between stages

    args:
    json: the parsed IDREM json file
    genenames: a list of genes in the nodes
    geneids: a list of gene ids in the nodes
    topN: the number of genes to accumulate gene weights
    total_stage: the total number of stages

    return:
    out: a list of top N up or down regulators of each path
    '''

    out = [[] for i in range(total_stage)]
    for i in range(total_stage):
        changegene = np.array([json[5][j] for j in geneids[i]])
        change = abs(changegene[:,i+1]-changegene[:,i])
        pddata = pd.DataFrame(change, columns=['change_Value'])
        pddata.index = genenames[i]
        sortedchange = pddata.sort_values(by = 'change_Value',ascending=False)
        topNGenes = sortedchange.index.tolist()[:topN]
        out[i] = topNGenes
    return out


# def matchTFandTG(TFs,scopes,filename,genenames):
#     '''
#     use target genes from IDREM as scopes to count tf and tgs with human-encode
#     '''
#     f = open('./'+filename,'r')
#     rl = f.readlines()
#     f.close()
#     TG = {}
#     for line in rl[1:]:
#         item = line.split('\t')
#         if item[0] not in TG.keys():
#             TG[item[0]] = []
#         genename = item[1].split(';')[0]
#         #if genename in genenames:
#         TG[item[0]].append(genename)
#     genedict = {}
#     for i, each in enumerate(genenames):
#         genedict[each.upper()] = i
#     TFTG = [[] for _ in range(len(TFs))]
#     for i, track in enumerate(TFs):
#         temp = [[] for _ in range(3)]
#         TFTG[i] = temp
        
#         for j, stage in enumerate(track):
#             for tf in stage:
                
#                 targetGenes = TG[tf[0].split(' ')[0]]
                
#                 temp2 = np.zeros(shape=len(genenames))
#                 for each in enumerate(targetGenes):
#                     if each[1] in genedict.keys() and each[1] in scopes[i][j]:
#                         temp2[genedict[each[1]]] = 1

#                 if tf[0].split(' ')[0] in genenames:
#                     index = genedict[tf[0].split(' ')[0]]
#                     temp2[index] = 2
#                 TFTG[i][j].append(temp2)
#             TFTG[i][j] = np.array(TFTG[i][j])
#             TFTG[i][j] = np.sum(TFTG[i][j],axis = 0)
#     TFTG = np.array(TFTG)
#     return TFTG

def listTracks(mid, iteration,total_stage):
    '''
    list all tracks in the selected iteration
    args:
    mid: directory to the task
    iteration: the selected iteration
    total_stage: the total number of stages

    return:
    tempTrack: a list of tracks
    '''
    filenames= os.listdir(os.path.join(mid,str(iteration)+'/idremResults/'))
    #filenames = os.listdir('./reresult/idremVizCluster0.1-nov14/') #defalut path
    tempTrack = [[] for _ in range(total_stage)]
    for each in filenames:
        temp = each.split('.')[0].split('-')
        for i,item in enumerate(temp):
            temp1 = item.split('n')
            tempTrack[i].append(temp1)
    return tempTrack

#a = listTracks('./reresult/idremVizCluster0.1-nov14')

# def updateGeneFactors(adata, clusterid,iteration,geneWeight):
#     geneWeight = geneWeight.reshape(1,-1)
#     adata.obs['leiden']=adata.obs['leiden'].astype('int64')
#     cells = adata.obs.reset_index()
#     celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
#     if 'geneWeight' not in adata.layers.keys():
#         adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
#     else:
#         adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
#     adata.layers['geneWeight'][celllist] += geneWeight
   
#     return adata

# def updataGeneTables(mid, iteration, geneFactors):
#     tracks = listTracks(mid,iteration)
#     difference = 0
#     for i, stage in enumerate(tracks):
#         if i != 0:
#             adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
#             previousMySigmoidGeneWeight = np.sum(mySigmoid(adata.layers['geneWeight'].toarray()))
#             #adata = sc.read_h5ad('./stagedata/%d.h5ad'%(i))
#             for j, clusterids in enumerate(stage):#in some tracks, the one stage have many clusters
#                 for clusterid in clusterids:
#                     # print('number of idrem file', j)
#                     # print('stage',i)
#                     adata = updateGeneFactors(adata,clusterid,iteration,geneFactors[j][i-1])
#                     # difference+=clusterDiff#each update will get the mySigmoid(increment) 
#             currentMySigmoidGeneWeight = np.sum(mySigmoid(adata.layers['geneWeight'].toarray()))
#             difference += currentMySigmoidGeneWeight - previousMySigmoidGeneWeight
#             adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
#             adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
#         else:
#             if int(iteration) == 0:
#                 adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
#                 if 'geneWeight' not in adata.layers.keys():
#                     adata.layers['geneWeight'] = csr_matrix(np.zeros(adata.X.shape)) 
#                     adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
#     return difference
def matchTFandTGWithFoldChange(TFs,scopes,avgCluster,filename,genenames,total_stage):
    '''
    use target genes from IDREM as scopes to count tf and tgs with human-encode
    tf factor 2
    target of tf in top dynamic genes 1
    target of tf not in top dynamic genes 0.5
    other genes 0
    '''
    f = open(filename,'r')
    rl = f.readlines()
    f.close()
    TG = {}
    for line in rl[1:]:
        item = line.split('\t')
        if item[0] not in TG.keys():
            TG[item[0]] = []
        genename = item[1].split(';')[0]
        #if genename in genenames:
        TG[item[0]].append(genename)
    genedict = {}
    for i, each in enumerate(genenames):
        genedict[each.upper()] = i
    TFTG = [[] for _ in range(len(TFs))]
    for i, track in enumerate(TFs):
        temp = [[] for _ in range(total_stage-1)]
        TFTG[i] = temp
        
        for j, stage in enumerate(track):
            TFTG[i][j] = np.zeros(shape=len(genenames))
            for tf in stage:
                
                targetGenes = TG[tf[0].split(' ')[0]]
                for each in scopes[i][j]:
                    foldChange = abs(np.log2(avgCluster[j+1][j][genedict[each]]+1) - np.log2(avgCluster[j][i][genedict[each]]+1))
                    TFTG[i][j][genedict[each]] = 1*foldChange+1
                temp2 = np.zeros(shape=len(genenames))
                for each in enumerate(targetGenes):
                    
                    
                    if each[1] in genedict.keys() and each[1] in scopes[i][j]:
                    
                    

                        foldChange = abs(np.log2(avgCluster[j+1][j][genedict[each[1]]]+1) - np.log2(avgCluster[j][i][genedict[each[1]]]+1))
                        # if TFTG[i][j][genedict[each[1]]]<2:
                        TFTG[i][j][genedict[each[1]]] = 1*foldChange+1 #hyperparameter needed to be adusted by users
                    elif each[1] in genedict.keys() and each[1] not in scopes[i][j]:
                        
                        foldChange = abs(np.log2(avgCluster[j+1][i][genedict[each[1]]]+1)-np.log2(avgCluster[j][i][genedict[each[1]]]+1))
                        
                        # if TFTG[i][j][genedict[each[1]]]<2:
                        TFTG[i][j][genedict[each[1]]] = 0.5*foldChange+0.5 #hyperparameter needed to be adusted by users

                if tf[0].split(' ')[0] in genenames:
                    index = genedict[tf[0].split(' ')[0]]
                    # if avgCluster[j+1][i][index] > avgCluster[j][i][index]:
                    #     if avgCluster[j][i][index] == 0:
                    #         foldChange = 1
                    #     else:
                    #         foldChange = avgCluster[j+1][i][index]/avgCluster[j][i][index]
                    # elif avgCluster[j+1][i][index] < avgCluster[j][i][index]:
                    #     if avgCluster[j+1][i][index] == 0:
                    #         foldChange = 1
                    #     else:
                    foldChange = abs(np.log2(avgCluster[j][i][index]+1)-np.log2(avgCluster[j+1][i][index]+1))
                    TFTG[i][j][index] = 2*foldChange+2
                #TFTG[i][j].append(temp2)
            #TFTG[i][j] = np.array(TFTG[i][j])
            #TFTG[i][j] = np.sum(TFTG[i][j],axis = 0)
    TFTG = np.array(TFTG)
    return TFTG

# def replaceGeneFactors(adata, clusterid,geneWeight):
#     '''
#     replace genefacotrs instead of accumulating and decaying
#     '''
#     geneWeight = geneWeight.reshape(1,-1)
#     adata.obs['leiden']=adata.obs['leiden'].astype('int64')
#     cells = adata.obs.reset_index()
#     celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
#     if 'geneWeight' not in adata.layers.keys():
#         adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
#     else:
#         adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
#     adata.layers['geneWeight'][celllist] = geneWeight
#     adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
#     return adata
def updateGeneFactorsWithDecay(adata, clusterid,iteration,geneWeight, decayRate = 0.5):
    '''
    update gene weights and decay the weight of genes that are not important in this iteration of a cluster

    args:
    adata: the cluster of single cell data
    clusterid: the cluster id
    iteration: the selected iteration
    geneWeight: the gene weights
    decayRate: the decay rate

    return:
    adata: the updated single-cell data
    '''
    geneWeight = geneWeight.reshape(1,-1)
    adata.obs['leiden']=adata.obs['leiden'].astype('int64')
    cells = adata.obs.reset_index()
    celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
    if 'geneWeight' not in adata.layers.keys():
        adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
    else:
        adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
    adata.layers['geneWeight'][celllist] = adata.layers['geneWeight'][celllist]*decayRate#.multiply(decayCandidate) change it to all decrease for all cells in one stage
    # adata.layers['geneWeight'][celllist] += geneWeight
    adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
    return adata
# def replaceGeneTables(mid, iteration, geneFactors):
#     '''
#     oct23, use gene table replacing instead of accumulating and decaying
#     '''
#     tracks = listTracks(mid,iteration)
#     difference = 0
#     for i, stage in enumerate(tracks):
#         if i != 0:
#             gc.collect()
#             adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
#             temppreviousMySigmoidGeneWeight = adata.layers['geneWeight']
#             for j, clusterids in enumerate(stage):#in some tracks, the one stage have many clusters
#                 for clusterid in clusterids:
#                     # print('number of idrem file', j)
#                     # print('stage',i)
#                     adata = replaceGeneFactors(adata,clusterid,geneFactors[j][i-1])
#             previousMySigmoidGeneWeight = mySigmoid(temppreviousMySigmoidGeneWeight.toarray())
#             currentMySigmoidGeneWeight = mySigmoid(adata.layers['geneWeight'].toarray())
#             difference += np.mean(np.absolute(currentMySigmoidGeneWeight - previousMySigmoidGeneWeight))
#             adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
#             adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
#         else:
#             if int(iteration) == 0:
#                 adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
#                 if 'geneWeight' not in adata.layers.keys():
#                     adata.layers['geneWeight'] = csr_matrix(np.zeros(adata.X.shape)) 
#                     adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
#     return difference/3       
def updataGeneTablesWithDecay(mid, iteration, geneFactors,total_stage, decayRate = 0.5):
    '''
    update gene weights and decay the weight of genes that are not important in this iteration

    args:
    mid: directory to the task
    iteration: the selected iteration
    geneFactors: the gene weights
    total_stage: the total number of stages
    decayRate: the decay rate

    return:
    difference: the average difference of gene weights between stages
    '''
    tracks = listTracks(mid,iteration,total_stage)
    difference = 0
    for i, stage in enumerate(tracks):
        if i != 0:
            gc.collect()
            adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
            temppreviousMySigmoidGeneWeight = adata.layers['geneWeight']
            adata.layers['geneWeight'] = adata.layers['geneWeight'].multiply(decayRate) #sep8 change to all decrease for all cell
            #adata = sc.read_h5ad('./stagedata/%d.h5ad'%(i))
            for j, clusterids in enumerate(stage):#in some tracks, the one stage have many clusters
                for clusterid in clusterids:
                    # print('number of idrem file', j)
                    # print('stage',i)
                    adata = updateGeneFactorsWithDecay(adata,clusterid,iteration,geneFactors[j][i-1])
                    
                    # difference+=clusterDiff#each update will get the mySigmoid(increment) 
            previousMySigmoidGeneWeight = mySigmoid(temppreviousMySigmoidGeneWeight.toarray())
            currentMySigmoidGeneWeight = mySigmoid(adata.layers['geneWeight'].toarray())
            difference += np.mean(np.absolute(currentMySigmoidGeneWeight - previousMySigmoidGeneWeight))
            adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
            adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
        else:
            if int(iteration) == 0:
                adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
                if 'geneWeight' not in adata.layers.keys():
                    adata.layers['geneWeight'] = csr_matrix(np.zeros(adata.X.shape)) 
                    adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
    return difference/(total_stage-1)
def checkupDown(drem, genename):
    '''
    check if the TFs is a up or down regulator
    args:
    drem: the DREM attribute
    genename: name of TF(str)
    
    return: if it is a up or down regulator return 1, else return 0
    '''
    genename=genename.split(' ')[0]

    if genename not in drem[3]:

        
        return 0
    else:
        return 1
# def filterUpandDown(node,stage,filename):
#     '''
#     filter TFs that aren't up or down regulators
#     args: 
#     node: 
#     stage:
#     filename: the name of 
    
#     return: 
#     filteredTFs: TFs that are up or down regulators
#     '''
#     filteredTFs=[]
#     genedict = upandDowndict(filename)
#     for each in node:
#         checker = checkupDown(genedict,each[0], stage)
#         if checker == 1:
#             filteredTFs.append(each)
#     return filteredTFs
def mergeTFs(TFs,total_stage):
    '''
    merge top N up or down regulators into the stage level and remove the repeated regulators among sibling nodes of IDREM tree
    args: 
    TFs: a list of top N up or down regulators of a IDREM tree
    total_stage: the total number of stages

    
    return: 
    out: a list of up or down regulators of each stage
    '''
    upAndDownset= [set(),set(),set()]
    out = [[] for _ in range(total_stage-1)]
    for i, each in enumerate(TFs):
        for item in each:
            for data in item:
                if data[0].split(' ')[0] not in upAndDownset[i]:
                    upAndDownset[i].add(data[0].split(' ')[0])
                    out[i].append(data)
    return out
    
def getTopNUpandDown(TFs,topN=20):
    '''
    obtain top 20 up or down regulators based on the score overall (P value)
    args: 
    TFs: a list of up or down regulators
    topN: the number of top regulators to be extracted. Default is 20
    
    return: 
    TFs top N up or down regulators
    '''
   
    if len(TFs[0]) == 6:
        TFs = sorted(TFs,key=lambda x:x[5])
    else:
        
        TFs = sorted(TFs,key=lambda x:x[6])
    return TFs[:topN]
def extractTFs(path,filename,total_stage,topN=20):
    '''
    extract top N up or down TFs of a certain path from the DREM json file
    args: 
    filename: the name of certain paths
    total_stage: the total number of stages
    topN: the number of top regulators to be extracted. Default is 20
    
    return:
    extractedTFs: top N up or down TFs of a certain path
    '''
    print('getting TFs from ', filename)
    path = os.path.join(path,filename,'DREM.json')

    extractedTFs = [[] for _ in range(total_stage-1)]
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    
    TFs = [[] for _ in range(total_stage-1)]
    stages = [str(i+1) for i in range(total_stage-1)]
    # stage1 = 'IPF1'
    # stage2 = 'IPF2'
    # stage3 = 'IPF3'
    for each in tt[0][1:]:
        temp = []
        for item in each['ETF']:
            if checkupDown(tt, item[0]):
                
                temp.append(item)
        if len(temp) ==0:
            continue
        temp = getTopNUpandDown(temp,topN)
        print(each['nodetime'],stages.index(each['nodetime']))
        TFs[stages.index(each['nodetime'])].append(temp)
        # if each['nodetime'] == stage1:
        #     TFs[0].append(temp)
        # elif each['nodetime'] == stage2: 
        #     TFs[1].append(temp)
        # elif each['nodetime'] == stage3:
        #     TFs[2].append(temp)
    
    extractedTFs = mergeTFs(TFs,total_stage=total_stage)
    
    return extractedTFs
def getTFs(path,total_stage,topN=20):
    '''
    get top N up or down regulators of each path
    args:
    path: the file path of IDREM results
    topN: the number of top regulators to be extracted. Default is 20
    total_stage: the total number of stages
    
    return:
    out: a list of top N up or down regulators of each path
    '''
    out = []
 
    filenames = os.listdir(path)
    
    for each in filenames:
        if each[0] != '.':
            out.append(extractTFs(path,each,topN=topN,total_stage=total_stage))
    return out
# def matchTG(TFs,filename,genenames):
#     f = open('./'+filename,'r')
#     rl = f.readlines()
#     f.close()
#     TG = {}
#     for line in rl[1:]:
#         item = line.split('\t')
#         if item[0] not in TG.keys():
#             TG[item[0]] = []
#         genename = item[1].split(';')[0]
#         #if genename in genenames:
#         TG[item[0]].append(genename)
#     genedict = {}
#     for i, each in enumerate(genenames):
#         genedict[each.upper()] = i
#     TFTG = [[] for _ in range(len(TFs))]
#     for i, track in enumerate(TFs):
#         temp = [[] for _ in range(3)]
#         TFTG[i] = temp
        
#         for j, stage in enumerate(track):
#             for tf in stage:
                
#                 targetGenes = TG[tf[0].split(' ')[0]]
                
#                 temp2 = np.zeros(shape=len(genenames))
#                 for each in enumerate(targetGenes):
#                     if each[1] in genedict.keys():
#                         temp2[genedict[each[1]]] = 1

#                 if tf[0].split(' ')[0] in genenames:
#                     index = genedict[tf[0].split(' ')[0]]
#                     temp2[index] = 2
#                 TFTG[i][j].append(temp2)
#             TFTG[i][j] = np.array(TFTG[i][j])
#             TFTG[i][j] = np.sum(TFTG[i][j],axis = 0)
#     TFTG = np.array(TFTG)
#     return TFTG

# def mySigmoid(z,weight=-0.5):
#     '''
#     shifted sigmoid transformation of given data
#     args: 
#     z: input data
#     return:  
#     out: data after shifted sigmoid transformation
#     '''
#     out = 2/(1+np.exp(weight*z))
#     return out

# def mySigmoid(z,weight=-4):
#     '''
#     new shifted sigmoid transformation for replace strategy
#     args: 
#     z: input data
#     return:  
#     out: data after shifted sigmoid transformation
#     '''
#     out = 1/(1+15*np.exp(weight*z))
#     return out

def mySigmoid(z,weight=-4):
    '''
    new shifted sigmoid transformation for replace strategy
    args: 
    z: input data
    return:  
    out: data after shifted sigmoid transformation
    '''
    out = 1/(1+20*np.exp(weight*z+1.5)) #hyper parameters needed to be adjusted by users
    return out

# def geneFactor(idrem_path, tfinfo_path, defaultfactor):
#     '''
#     get use shifted sigmoid to normalize tf info and obtain the gene factors for retraining

#     args:
#     path to tf info file
#     return: 
#     out: gene factor matrics
#     '''
#     filenames = os.listdir(idrem_path)
#     stage = [[] for _ in range(4)]
#     for each in filenames:
#         temp = each.split('.')[0].split('-')
#         for i,item in enumerate(temp):
#             temp1 = item.split('n')
#             stage[i].append(temp1)
#     genes =np.load(tfinfo_path, allow_pickle=True)
#     #out = [{} for _ in range(10)]

#     for i in range(len(genes)):
#         for j in range(len(genes[i])):
#             for each in stage[j+1][i]:
#                 defaultfactor[j+1][each] = mySigmoid(np.sum(genes[i][j],axis=0))
#             #defaultfactor[i].append(mySigmoid(np.sum(genes[i][j],axis=0)))
#     return defaultfactor
    
if __name__ == '__main__':
    getPosNegDynamicPathGenes('/mnt/md0/yumin/to_upload/UNAGI/tutorials/example_1/idrem','4-4-5-1.txt_viz',50)