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
def getMinMaxPathID(json):
    maxid = -1
    minid = -1
    maxval = 0
    minval = 99
    for each in json[0][1:]:
        if each['nodetime'] == 'IPF3':
            if each['nodeMean']>maxval:
                maxval = each['nodeMean']
                maxid = each['nodeID']
            elif each['nodeMean']<minval:
                minval = each['nodeMean']
                minid = each['nodeID']
    return maxid,minid
def getMinMaxNodeID(json):
    maxid = [-99 for i in range(3)]
    minid = [-99 for i in range(3)]
    maxval = [-99 for i in range(3)]
    minval = [99 for i in range(3)]
    for each in json[0]:
        if each['nodetime'] == 'IPF3':
            if each['nodeMean']>maxval[2]:
                maxval[2] = each['nodeMean']
                maxid[2] = each['nodeID']
            if each['nodeMean']<minval[2]:
                minval[2] = each['nodeMean']
                minid[2] = each['nodeID']
        elif each['nodetime'] == 'IPF2':
            if each['nodeMean']>maxval[1]:
                maxval[1] = each['nodeMean']
                maxid[1] = each['nodeID']
            if each['nodeMean']<minval[1]:
                minval[1] = each['nodeMean']
                minid[1] = each['nodeID']
        elif each['nodetime'] == 'IPF1':
            if each['nodeMean']>maxval[0]:
                maxval[0] = each['nodeMean']
                maxid[0] = each['nodeID']
            if each['nodeMean']<minval[0]:
                minval[0] = each['nodeMean']
                minid[0] = each['nodeID']
    return maxid,minid
def getMinMaxNodeID0(json):
    maxid = -1
    minid = -1
    maxval = 0
    minval = 99
    for each in json[0][1:]:
        if each['nodetime'] == 'IPF3':
            if each['nodeMean']>maxval:
                maxval = each['nodeMean']
                maxid = each['nodeID']
            elif each['nodeMean']<minval:
                minval = each['nodeMean']
                minid = each['nodeID']
    return maxid,minid
def getStage3IDs(json,total_stage):
    '''
    get the list of all last stage node ids
    args:
    json: idrem meta
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
    
    '''
    
    stage3IDs = getStage3IDs(json,total_stage) #stage3IDs: the list of all IPF3 node ids
    paths = []
    for i in stage3IDs:
        paths.append(getidremPath(json, i))
    return paths
def getPosNegMaxChangesNodes(json,paths):
    means = []
    for path in paths:
        pathmean = []
        for each in path:
            
            for node in json[0][1:]:
                if node['nodeID'] == each:
                    pathmean.append(node['nodeMean'])
        means.append(pathmean)
    means = np.array(means)
    means[:,2] = means[:,2]-means[:,1]
    means[:,1] = means[:,1]-means[:,0]
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
    tt = readIdremJson(path, filename)
    paths = getIdremPaths(tt)
    return getPosNegMaxChangesNodes(tt,paths)


def getPosNegDynamicPathGenes(path,filename,topN):
    '''
    get the genes from nodes that increase and decrease most between stages
    '''
    total_stage = len(filename.split('.')[0].split('-'))
    tt = readIdremJson(path, filename)
    paths = getIdremPaths(tt,total_stage)
    posdynamicids, negdynamicids = getPosNegMaxChangesNodes(tt,paths)
    posdynamicgenes,posdynamicgeneids = getMaxOrMinNodesGenes(tt,posdynamicids)
    negdynamicgenes,negdynamicgeneids = getMaxOrMinNodesGenes(tt,negdynamicids)
    posnegdynamicgenes = [posdynamicgenes[i]+negdynamicgenes[i] for i in range(3)]
    posnegdynamicgeneids = [posdynamicgeneids[i]+negdynamicgeneids[i] for i in range(3)]
    out = getTopNTargetGenes(tt,posnegdynamicgenes,posnegdynamicgeneids,topN)
    return out


def getMinMaxNode(json,maxid,minid):
    maxnodes = getMinOrMaxPath0(json,maxid)
    minnodes = getMinOrMaxPath0(json,minid)
    return maxnodes,minnodes
def getMinMaxPath(json,maxid,minid):
    maxnodes = getMinOrMaxPath(json,maxid)
    minnodes = getMinOrMaxPath(json,minid)
    return maxnodes,minnodes
def getMinOrMaxPath(json,cids):
    nodes = []
    for i, cid in enumerate(cids):
        for each in reversed(json):
            if each['nodeID'] == cid:
                nodes.append(cid)
              
    return nodes
def getMinOrMaxPath0(json,cid):
    nodes = []
    for each in reversed(json[1:]):
        if each['nodeID'] == cid:
            nodes.append(cid)
            if each['parent']!= -1:
                cid = each['parent']
            else:
                break
    return nodes
def getMaxOrMinNodesGenes(json,nodes):
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
def getMaxOrMinNodesGenes0(json,nodes):
    boolgenes = []
    genes = []
    geneids = []
    for each in json[0][1:]:
        if each['nodeID'] in nodes:
            boolgenes.append(each['genesInNode'])
    for each in boolgenes:

        for i, gene in enumerate(each):
            if gene == True:
                genes.append(json[3][i])
                geneids.append(i)
    genes = list(set(genes))
    geneids = list(set(geneids))
    return genes,geneids
def getMaxOrMinPathGenes(json,nodes):
    boolgenes = []
    genes = []
    geneids = []
    for each in json[0][1:]:
        if each['nodeID'] in nodes:
            boolgenes.append(each['genesInNode'])
    for each in boolgenes:

        for i, gene in enumerate(each):
            if gene == True:
                genes.append(json[3][i])
                geneids.append(i)
    genes = list(set(genes))
    geneids = list(set(geneids))
    return genes,geneids
def readIdremJson(path, filename):
    print('getting Target genes from ', filename)
    path = os.path.join(path,filename,'DREM.json')
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    return tt
def getMaxMinPathGenes(path,filename,topN):
    tt = readIdremJson(path, filename)
    maxid,minid = getMinMaxNodeID(tt)
    # maxnodes,minnodes = getMinMaxPath(tt[0],maxid,minid)
    maxgenes,maxgeneids = getMaxOrMinNodesGenes(tt,maxid)
    mingenes,mingeneids = getMaxOrMinNodesGenes(tt,minid)
    maxmingenes = [maxgenes[i]+mingenes[i] for i in range(3)]
    maxmingeneids = [maxgeneids[i]+mingeneids[i] for i in range(3)]
    # maxmingenes = [list(set(maxmingenes[i])) for i in range(3)]
    # maxmingeneids = [list(set(maxmingeneids[i])) for i in range(3)]
    
    out = getTopNTargetGenes(tt,maxmingenes,maxmingeneids,topN)
    return out
def getMaxMinPathGenes0(path,filename,topN):
    tt = readIdremJson(path, filename)
    maxid,minid = getMinMaxNodeID0(tt)
    maxnodes,minnodes = getMinMaxNode(tt[0],maxid,minid)
    
    maxgenes,maxgeneids = getMaxOrMinNodesGenes0(tt,maxnodes)
    mingenes,mingeneids = getMaxOrMinNodesGenes0(tt,minnodes)
    maxmingenes = maxgenes+mingenes
    maxmingeneids = maxgeneids+mingeneids
    maxmingenes = list(set(maxmingenes))
    
    maxmingeneids = list(set(maxmingeneids))
    out = getTopNTargetGenes0(tt,maxmingenes,maxmingeneids,topN)
    return out
def getTopNTargetGenes(json,genenames,geneids,topN):

    out = [[] for i in range(3)]
    for i in range(3):
        changegene = np.array([json[5][j] for j in geneids[i]])
        change = abs(changegene[:,i+1]-changegene[:,i])
        pddata = pd.DataFrame(change, columns=['change_Value'])
        pddata.index = genenames[i]
        sortedchange = pddata.sort_values(by = 'change_Value',ascending=False)
        topNGenes = sortedchange.index.tolist()[:topN]
        out[i] = topNGenes
    return out
def getTopNTargetGenes0(json,genenames,geneids,topN):
    out = [[] for i in range(3)]
    for i in range(3):
        changegene = np.array([json[5][j] for j in geneids])
        change = abs(changegene[:,i+1]-changegene[:,i])
        pddata = pd.DataFrame(change, columns=['change_Value'])
        pddata.index = genenames
        sortedchange = pddata.sort_values(by = 'change_Value',ascending=False)
        topNGenes = sortedchange.index.tolist()[:topN]
        out[i] = topNGenes
    return out
#def getTopNTargetGenesUnion():
def getTargetGenes0(path,N):
    '''
    get top N genes of each path
    args:
    path: the file path of IDREM results
    
    return:
    out: a list of top 20 up or down regulators of each path
    '''
    out = []
 
    filenames = os.listdir(path)
    
    for each in filenames:
        if each[0] != '.':
            out.append(getMaxMinPathGenes0(path,each,N))
    return out    
def getTargetGenes(path,N):
    '''
    get top N genes of each path
    args:
    path: the file path of IDREM results
    
    return:
    out: a list of top 20 up or down regulators of each path
    '''
    out = []
 
    filenames = os.listdir(path)
    
    for each in filenames:
        if each[0] != '.':
            #out.append(getMaxMinPathGenes(path,each,N)) #get the genes from nodes that have highest and lowest nodemean
            out.append(getPosNegDynamicPathGenes(path,each,N)) #get the genes from nodes increase and descrease most between stages
    return out
def matchTFandTG(TFs,scopes,filename,genenames):
    '''
    use target genes from IDREM as scopes to count tf and tgs with human-encode
    '''
    f = open('./'+filename,'r')
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
        temp = [[] for _ in range(3)]
        TFTG[i] = temp
        
        for j, stage in enumerate(track):
            for tf in stage:
                
                targetGenes = TG[tf[0].split(' ')[0]]
                
                temp2 = np.zeros(shape=len(genenames))
                for each in enumerate(targetGenes):
                    if each[1] in genedict.keys() and each[1] in scopes[i][j]:
                        temp2[genedict[each[1]]] = 1

                if tf[0].split(' ')[0] in genenames:
                    index = genedict[tf[0].split(' ')[0]]
                    temp2[index] = 2
                TFTG[i][j].append(temp2)
            TFTG[i][j] = np.array(TFTG[i][j])
            TFTG[i][j] = np.sum(TFTG[i][j],axis = 0)
    TFTG = np.array(TFTG)
    return TFTG

def listTracks(mid, iteration):
    filenames= os.listdir(os.path.join(mid,str(iteration)+'/idremResults/'))
    #filenames = os.listdir('./reresult/idremVizCluster0.1-nov14/') #defalut path
    tempTrack = [[] for _ in range(4)]
    for each in filenames:
        temp = each.split('.')[0].split('-')
        for i,item in enumerate(temp):
            temp1 = item.split('n')
            tempTrack[i].append(temp1)
    return tempTrack

#a = listTracks('./reresult/idremVizCluster0.1-nov14')

def updateGeneFactors(adata, clusterid,iteration,geneWeight):
    geneWeight = geneWeight.reshape(1,-1)
    adata.obs['leiden']=adata.obs['leiden'].astype('int64')
    cells = adata.obs.reset_index()
    celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
    if 'geneWeight' not in adata.layers.keys():
        adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
    else:
        adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
    adata.layers['geneWeight'][celllist] += geneWeight
   
    return adata

def updataGeneTables(mid, iteration, geneFactors):
    tracks = listTracks(mid,iteration)
    difference = 0
    for i, stage in enumerate(tracks):
        if i != 0:
            adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
            previousMySigmoidGeneWeight = np.sum(mySigmoid(adata.layers['geneWeight'].toarray()))
            #adata = sc.read_h5ad('./stagedata/%d.h5ad'%(i))
            for j, clusterids in enumerate(stage):#in some tracks, the one stage have many clusters
                for clusterid in clusterids:
                    print('number of idrem file', j)
                    print('stage',i)
                    adata = updateGeneFactors(adata,clusterid,iteration,geneFactors[j][i-1])
                    # difference+=clusterDiff#each update will get the mySigmoid(increment) 
            currentMySigmoidGeneWeight = np.sum(mySigmoid(adata.layers['geneWeight'].toarray()))
            difference += currentMySigmoidGeneWeight - previousMySigmoidGeneWeight
            adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
            adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
        else:
            if int(iteration) == 0:
                adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
                if 'geneWeight' not in adata.layers.keys():
                    adata.layers['geneWeight'] = csr_matrix(np.zeros(adata.X.shape)) 
                    adata.write(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%i),compression='gzip' )
    return difference
def matchTFandTGWithFoldChange(TFs,scopes,avgCluster,filename,genenames):
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
        temp = [[] for _ in range(3)]
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
                        TFTG[i][j][genedict[each[1]]] = 1*foldChange+1
                    elif each[1] in genedict.keys() and each[1] not in scopes[i][j]:
                        
                        foldChange = abs(np.log2(avgCluster[j+1][i][genedict[each[1]]]+1)-np.log2(avgCluster[j][i][genedict[each[1]]]+1))
                        
                        # if TFTG[i][j][genedict[each[1]]]<2:
                        TFTG[i][j][genedict[each[1]]] = 0.5*foldChange+0.5

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

def replaceGeneFactors(adata, clusterid,geneWeight):
    '''
    replace genefacotrs instead of accumulating and decaying
    '''
    geneWeight = geneWeight.reshape(1,-1)
    adata.obs['leiden']=adata.obs['leiden'].astype('int64')
    cells = adata.obs.reset_index()
    celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
    if 'geneWeight' not in adata.layers.keys():
        adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
    else:
        adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
    adata.layers['geneWeight'][celllist] = geneWeight
    adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
    return adata
def updateGeneFactorsWithDecay(adata, clusterid,iteration,geneWeight, decayRate = 0.5):
    ##aug 12 change this to all decrease
#     decayCandidate = []
#     for each in geneWeight:
#         if each>0:
#             decayCandidate.append(1)
#         else:
#             decayCandidate.append(decayRate)
#     decayCandidate = np.array(decayCandidate)
    geneWeight = geneWeight.reshape(1,-1)
    adata.obs['leiden']=adata.obs['leiden'].astype('int64')
    cells = adata.obs.reset_index()
    celllist = cells[cells['leiden']==int(clusterid)].index.tolist()
    if 'geneWeight' not in adata.layers.keys():
        adata.layers['geneWeight'] = lil_matrix(np.zeros(adata.X.shape)) 
    else:
        adata.layers['geneWeight'] = adata.layers['geneWeight'].tolil()
    #adata.layers['geneWeight'][celllist] = adata.layers['geneWeight'][celllist]*decayRate#.multiply(decayCandidate) change it to all decrease for all cells in one stage
    adata.layers['geneWeight'][celllist] += geneWeight
    adata.layers['geneWeight'] = adata.layers['geneWeight'].tocsr()
    return adata
def replaceGeneTables(mid, iteration, geneFactors):
    '''
    oct23, use gene table replacing instead of accumulating and decaying
    '''
    tracks = listTracks(mid,iteration)
    difference = 0
    for i, stage in enumerate(tracks):
        if i != 0:
            gc.collect()
            adata = sc.read_h5ad(os.path.join(mid,str(iteration)+'/stagedata/%d.h5ad'%(i)))
            temppreviousMySigmoidGeneWeight = adata.layers['geneWeight']
            for j, clusterids in enumerate(stage):#in some tracks, the one stage have many clusters
                for clusterid in clusterids:
                    print('number of idrem file', j)
                    print('stage',i)
                    adata = replaceGeneFactors(adata,clusterid,geneFactors[j][i-1])
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
    return difference/3       
def updataGeneTablesWithDecay(mid, iteration, geneFactors, decayRate = 0.5):
    tracks = listTracks(mid,iteration)
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
                    print('number of idrem file', j)
                    print('stage',i)
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
    return difference/3
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
def filterUpandDown(node,stage,filename):
    '''
    filter TFs that aren't up or down regulators
    args: 
    node: 
    stage:
    filename: the name of 
    
    return: 
    filteredTFs: TFs that are up or down regulators
    '''
    filteredTFs=[]
    genedict = upandDowndict(filename)
    for each in node:
        checker = checkupDown(genedict,each[0], stage)
        if checker == 1:
            filteredTFs.append(each)
    return filteredTFs
def mergeTFs(TFs):
    '''
    merge top 20 up or down regulators into the stage level and remove the repeated regulators among sibling nodes of IDREM tree
    args: a list of top 20 up or down regulators of a IDREM tree
    
    return: 
    out: a list of up or down regulators of each stage
    '''
    upAndDownset= [set(),set(),set()]
    out = [[] for _ in range(3)]
    for i, each in enumerate(TFs):
        for item in each:
            for data in item:
                if data[0].split(' ')[0] not in upAndDownset[i]:
                    upAndDownset[i].add(data[0].split(' ')[0])
                    out[i].append(data)
    return out
    
def getTop20UpandDown(TFs):
    '''
    obtain top 20 up or down regulators based on the score overall (P value)
    args: 
    TFs: a list of up or down regulators
    
    return: 
    TFs[:20]: top 20 up or down regulators
    '''
   
    if len(TFs[0]) == 6:
        TFs = sorted(TFs,key=lambda x:x[5])
    else:
        
        TFs = sorted(TFs,key=lambda x:x[6])
    return TFs[:20]
def extractTFs(path,filename):
    '''
    extract top 20 up or down TFs of a certain path from the DREM json file
    args: 
    filename: the name of certain paths
    
    return:
    extractedTFs: top 20 up or down TFs of a certain path
    '''
    print('getting TFs from ', filename)
    path = os.path.join(path,filename,'DREM.json')

    extractedTFs = [[] for _ in range(3)]
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    
    TFs = [[] for _ in range(3)]
    stage1 = 'IPF1'
    stage2 = 'IPF2'
    stage3 = 'IPF3'
    for each in tt[0][1:]:
        temp = []
        for item in each['ETF']:
            if checkupDown(tt, item[0]):
                
                temp.append(item)
        if len(temp) ==0:
            continue
        temp = getTop20UpandDown(temp)
        
        if each['nodetime'] == stage1:
            TFs[0].append(temp)
        elif each['nodetime'] == stage2: 
            TFs[1].append(temp)
        elif each['nodetime'] == stage3:
            TFs[2].append(temp)
    
    extractedTFs = mergeTFs(TFs)
    
    return extractedTFs
def getTFs(path):
    '''
    get top 20 up or down regulators of each path
    args:
    path: the file path of IDREM results
    
    return:
    out: a list of top 20 up or down regulators of each path
    '''
    out = []
 
    filenames = os.listdir(path)
    
    for each in filenames:
        if each[0] != '.':
            out.append(extractTFs(path,each))
    return out
def matchTG(TFs,filename,genenames):
    f = open('./'+filename,'r')
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
        temp = [[] for _ in range(3)]
        TFTG[i] = temp
        
        for j, stage in enumerate(track):
            for tf in stage:
                
                targetGenes = TG[tf[0].split(' ')[0]]
                
                temp2 = np.zeros(shape=len(genenames))
                for each in enumerate(targetGenes):
                    if each[1] in genedict.keys():
                        temp2[genedict[each[1]]] = 1

                if tf[0].split(' ')[0] in genenames:
                    index = genedict[tf[0].split(' ')[0]]
                    temp2[index] = 2
                TFTG[i][j].append(temp2)
            TFTG[i][j] = np.array(TFTG[i][j])
            TFTG[i][j] = np.sum(TFTG[i][j],axis = 0)
    TFTG = np.array(TFTG)
    return TFTG

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
def transfer_to_ranking_score(gw):
    '''
    ranking score
    '''
    # gw = adata.layers['geneWeight'].toarray()
    od = gw.shape[1]-rankdata(gw,axis=1)+1
    score = 1+1/np.power(od,0.5)
    
    return score

def mySigmoid(z,weight=-4):
    '''
    new shifted sigmoid transformation for replace strategy
    args: 
    z: input data
    return:  
    out: data after shifted sigmoid transformation
    '''
    out = 1/(1+20*np.exp(weight*z+1.5))
    return out
def geneFactor(tfinfo_path):
    '''
    get use shifted sigmoid to normalize tf info and obtain the gene factors for retraining
    
    args:
    path to tf info file
    return: 
    out: gene factor matrics
    '''
    genes =np.load(tfinfo_path,allow_pickle=True)
    out = [[] for _ in range(10)]

    for i in range(len(genes)):
        for j in range(len(genes[i])):
        
            out[i].append(mySigmoid(np.sum(genes[i][j],axis=0)))
    return out
def geneFactor(idrem_path, tfinfo_path, defaultfactor):
    '''
    get use shifted sigmoid to normalize tf info and obtain the gene factors for retraining

    args:
    path to tf info file
    return: 
    out: gene factor matrics
    '''
    filenames = os.listdir(idrem_path)
    stage = [[] for _ in range(4)]
    for each in filenames:
        temp = each.split('.')[0].split('-')
        for i,item in enumerate(temp):
            temp1 = item.split('n')
            stage[i].append(temp1)
    genes =np.load(tfinfo_path, allow_pickle=True)
    #out = [{} for _ in range(10)]

    for i in range(len(genes)):
        for j in range(len(genes[i])):
            for each in stage[j+1][i]:
                defaultfactor[j+1][each] = mySigmoid(np.sum(genes[i][j],axis=0))
            #defaultfactor[i].append(mySigmoid(np.sum(genes[i][j],axis=0)))
    return defaultfactor