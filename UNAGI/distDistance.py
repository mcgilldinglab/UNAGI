import multiprocessing
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
def runGetSimilarity(adata):
    ids = []
    for i in range(4):

        ids.append(adata.obs[adata.obs['stage'] == i].index.tolist())
    out = [[] for _ in range(4)]
    for i in range(4):
        temp = adata[ids[i]]
        #temp2 = adata[ids[i+1]]
        out[i]= getN2CSimilarity(temp, i)

    return out
def getN2CSimilarity(adata,stagei):
    '''
    calculate top differential gene similarity between two clusters in a stage
    '''
    
    clusterIds = set(adata.obs['leiden'].values)
    
    print(clusterIds)
    #maxj = adataj.obs['leiden'].max()
    similarities = {}
    for i,clusterIdi in enumerate(clusterIds):
        similarities[str(clusterIdi)] = []
        for j,clusterIdj in enumerate(clusterIds):
            similarities[str(clusterIdi)].append(getSimilarity(adata.uns['topGene'][str(stagei)],adata.uns['topGene'][str(stagei)],i,j))

    return similarities
def getSimilarity(adatai,adataj, i, j):
    '''
    calculate the differential gene similarity between two clusters. Differentail gene similarity = (1-Jaccard index)* gene ranking difference factor
    args: 
    adatai: anndata of stage i
    adataj: anndata of stage j
    i: the cluster id in stage i
    j: the cluster id in stage j

    return: 
    distance: the distance of top 100 differential gene between two clusters

    '''
    #iDE = adatai.d1.raw.var['features'][adatai.d1.uns['rank_genes_groups']['names'][str(i)]].values
    #jDE = adataj.d1.raw.var['features'][adataj.d1.uns['rank_genes_groups']['names'][str(j)]].values
    if isinstance(adatai[i], list):
        
        iDE = adatai[i]
    else:
        iDE = adatai[i].tolist()
    if isinstance(adataj[j], list):
        
        jDE = adataj[j]
    else:
        jDE = adataj[j].tolist()
  

    intersection = 0
    orderdistance = 0
    for i, each in enumerate(iDE):
        if each in jDE:
            
            intersection+=1
            jindex = jDE.index(each)
        
            orderdistance += abs(jindex-i)
        else:
            orderdistance+=100
    union = len(iDE)+len(jDE) - intersection
    JaccardIndex = intersection/union
    # print(intersection,union)
   
    orderdistance = orderdistance/(100*len(jDE))
    # print(orderdistance)

    return (1-JaccardIndex)*orderdistance
def normalizeDistanceRE(distance):#good
    '''
    Normalize the kl divergence distance and top differential gene distances. (Use the z-score method.)
    
    args: 
    distance: a list of distance metrics. ([gaussian kl divergence, top differential gene difference])
    
    return:
    normalizedDistance: sum of normalized kl divergence and top differential gene difference distance
    '''
    
    distance = np.array(distance)
    
    gaussiankld = np.array(distance[:,0].tolist()).reshape(-1,)
    
    #gammakld = np.array(distance[:,1].tolist()).reshape(-1,)

    de = np.array(distance[:,1].tolist())

    
    gaussian_kl_zscore = gaussiankld#stats.zscore(gaussiankld)
    #gamma_kl_zscore = stats.zscore(gammakld)
    de_zscore = stats.zscore(de)
    #ggkld = np.array(gaussiankld)+np.array(gammakld)
    #kld_zscore = stats.zscore(ggkld)
    return gaussian_kl_zscore+de_zscore#gaussian_kl_zscore+gamma_kl_zscore+de_zscore

def calculateN2CKLDistance(mu,sigma,previousRep,topGeneD):
    '''
    calculate the distance between a cell and all clusters in the same stage
    args: mu, sigma is the attribute of one cell
    previousRep: previous representation of clusters in one stage [#cluster,[gaussian(3 parameters)*#hidden]
    topGeneD: top gene distances between two clusters of one stage
    '''
    mean1 = mu
    std1 = sigma
    #mean1 = cluster1[:,0]
    covariance1 = torch.tensor(np.diag(std1**2))
    p = MultivariateNormal(torch.tensor(mean1), covariance1)
    d = []

    for no, each in enumerate(previousRep):
        
        cluster2 = np.array(each[0])
        mean2 = cluster2[:,0]
        std2 = cluster2[:,1]
        covariance2 = torch.tensor(np.diag(std2**2))
        q = MultivariateNormal(torch.tensor(mean2), covariance2)
        kl = torch.distributions.kl.kl_divergence(p, q)



        d.append([kl.detach().numpy(),topGeneD[no]])
    out = normalizeDistanceRE(d)
    return out

class calculateN2CKLDProcess(multiprocessing.Process):
    def __init__(self, mu,sigma,previousRep,topGeneD,ddd,tempids,idnumber,currentCluster,out,cellid):
        multiprocessing.Process.__init__(self)
        self.mu = mu
        self.sigma = sigma
        self.previousRep = previousRep
        self.topGeneD = topGeneD
        self.ddd = ddd
        self.tempids = tempids
        self.idnumber = idnumber
        self.currentCluster = currentCluster
        self.out = out
    def run(self):
        #time.sleep(10)
        time1 = time.time()
        temp = calculateN2CKLDistance(self.mu,self.sigma,self.previousRep,self.topGeneD)
        time2 = time.time()
        print(time2-time1)
        idx = np.argmin(np.array(temp))
        self.out[cellid] = [idx, self.idnumber]
        #self.out[]
        #if idx != int(self.currentCluster):
        #    self.ddd.append(idx)
        #    self.tempids.append(self.idnumber)
def mcSampling(mus, sigmas):
    '''
    monte carlo sampling strategy to sample data points from a gaussian distribution in each cell,
    for example the hidden space is 10, then sample 100 data point from input cell. If the number of cell is 200, then
    the sampled matrix will be [10, 200*100]
    args:
    mus, sigmas: mu and sigma vectors of input cells (shape: [number of cell, number of hidden nodes])
    
    return:
    samplegaussian: a list of sampled gaussian datapoints
    '''
    samplegaussian= [[] for _ in range(64)]
    hiddensize=64#len(mus[0])
    mus = np.vstack(mus)
    sigmas=np.vstack(sigmas)

    for i in range(hiddensize):
        
        mean = mus[:,i]
        mean = mean.repeat(100)
        std = sigmas[:,i]
        std = std.repeat(100)
        normal = Normal(torch.zeros(size=mean.shape)+torch.tensor(mean),torch.zeros(size=std.shape)+torch.tensor(std)).sample()
        samplegaussian[i].append(normal.numpy())
        '''for j in range(len(mus[i])):
            
            mean = mus[i][j]
            theta = thetas[i][j]
            std = np.exp(sigmas[i][j] / 2)
            #print(theta)
            normal = Normal(torch.zeros((2))+torch.tensor(mean),torch.zeros((2))+torch.tensor(std)).sample()
            gamma = Gamma(torch.ones(2), torch.zeros(2)+1/theta).sample()
#minimal unit test 100->2
            samplegamma[j].extend(gamma.numpy())
            samplegaussian[j].extend(normal.numpy())'''

    return samplegaussian

class GaussianRepThread(threading.Thread):
    def __init__(self, output, data,i):
        threading.Thread.__init__(self)
        self.i = i
        self.data = data
        self.output = output
    def run(self):
        #time.sleep(10)
        self.output[self.i] = norm.fit(self.data)
def fitClusterGaussianRepresentation(data):
    '''
    fit gaussian distributions for each hidden node
    args: 
    data: samples of the gaussian distribution of each hidden node
    return:
    out: list of mu and sigma of each hidden node
    '''
    threads = []
    out = [[] for _ in range(len(data))]
    for i in range(len(data)):
        threads.append(GaussianRepThread(out,data[i],i))
    for each in threads:
        each.start()
    for each in threads:
        each.join()
    return out
class GammaRepThread(threading.Thread):
    def __init__(self, output, data,i):
        threading.Thread.__init__(self)
        self.i = i
        self.data = data
        self.output = output
    def run(self):
        #time.sleep(10)
        self.output[self.i] = gamma.fit(self.data)
def fitClusterGammaRepresentation(data):
    '''
    Fit gamma distributions for each hidden node
    args: 
    data: samples of the gamma distribution of each hidden node
    return:
    out: list of alpha, beta of each hidden node
    '''
    threads = []
    out = [[] for _ in range(len(data))]
    for i in range(len(data)):

        
        threads.append(GammaRepThread(out,data[i][0],i))
    for each in threads:
        each.start()
    for each in threads:
        each.join()
    return out

def getClusterRepresentation(mus, sigmas):
    '''
    MC strategy to sample gaussian data points. Use sampled data points to fit gaussian distributions
    args: 
    mus, sigmas: mu and sigma vectors of input cells (shape: [number of cell, number of hidden nodes])
   
    return: 
    fitClusterGaussianRepresentation(sampling[0]):
    a list of fitted gaussian distributions
    '''
    T1 = time.time()
    sampling = mcSampling(mus,sigmas)
    T2 = time.time()
    #print('time is %s ' %((T2-T1)))
    return fitClusterGaussianRepresentation(sampling)
def clustertype(adata):
    '''
    find the most common cell types to represent the cluster
    args:
    adata: anndata of one cluster
    
    return: the most common cell types in the cluster
    '''
    dic = {}
    for each in adata.obs['cell.type.ident.fine']:
        if each not in dic.keys():
            dic[each]=1
        else:
            dic[each]+=1
    return max(dic, key=dic.get)
def normalizeDistance(distance):
    '''
    Normalize the kl divergence distance and top differential gene distances. (Use the z-score method.)
    
    args: 
    distance: a list of distance metrics. ([gaussian kl divergence, top differential gene difference])
    
    return:
    normalizedDistance: sum of normalized kl divergence and top differential gene difference distance
    '''
    

    distance = np.array(distance)
    
    gaussiankld = np.array(distance[:,0])

    de = np.array(distance[:,1])
    if gaussiankld.shape[0] != 1:

        gaussian_kl_zscore = stats.zscore(gaussiankld)
        #gamma_kl_zscore = stats.zscore(gammakld)
        de_zscore = stats.zscore(de)
        ggkld = gaussiankld
        ggkld_zscore = stats.zscore(ggkld)
    else:
        ggkld_zscore= gaussiankld
        de_zscore = de
    
    
    return ggkld_zscore+de_zscore#gaussian_kl_zscore+gamma_kl_zscore+de_zscore
def calculateKL(cluster1gaussian, cluster2gaussian):
    '''
    calculate KL divergence of multivariate gaussian distributions between two clusters.
    args: 
    cluster1gaussian: a list of [mean, std] of gaussian distribution of cluster 1

    cluster2gaussian:a list of [mean, std] of gaussian distribution of cluster 2

    
    return: kl divergence of multivariate gaussian distributions
    
    '''
    cluster1 = np.array(cluster1gaussian).reshape(-1,2)
    cluster2 = np.array(cluster2gaussian).reshape(-1,2)
    
    std1 = cluster1[:,1]
    covariance1 = torch.tensor(np.diag(std1**2))

    mean1 = cluster1[:,0]

    p = MultivariateNormal(torch.tensor(mean1), covariance1)
    mean2 = cluster2[:,0]
    std2 = cluster2[:,1]
    covariance2 = torch.tensor(np.diag(std2**2))
    q = MultivariateNormal(torch.tensor(mean2), covariance2)
    kl = torch.distributions.kl.kl_divergence(p, q)
    
  

    return kl.detach().numpy()

import time



