import multiprocessing
import numpy as np
import gc
import anndata
import time
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

def getSimilarity(adatai,adataj, i, j):
    '''
    calculate the differential gene similarity between two clusters. Differentail gene similarity = (1-Jaccard index) * gene ranking difference factor
    
    parameters
    -------------------
    adatai: anndata
        The data of stage i
    adataj: anndata 
        The data of stage j
    i: int
        the cluster id in stage i
    j: int
        the cluster id in stage j

    return
    -------------------
    distance: np.array
        The distance of top 100 differential gene between two clusters

    '''
  
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
   
    orderdistance = orderdistance/(100*len(jDE))
    # print(orderdistance)
    DEG_distance = (1-JaccardIndex)*orderdistance
    return DEG_distance

def mcSampling(mus, sigmas):
    '''
    monte carlo sampling strategy to sample data points from a gaussian distribution in each cell,
    for example the hidden space is 10, then sample 100 data point from input cell. If the number of cell is 200, then
    the sampled matrix will be [10, 200*100]
    
    parameters
    -------------------
    mus: list
        mu vectors of input cells (shape: [number of cell, number of hidden nodes])

    sigmas: list
        sigma vectors of input cells (shape: [number of cell, number of hidden nodes])

    
    return
    -------------------
    samplegaussian: list
        A list of sampled data-points from fitted gaussian distributions.
    '''
    samplegaussian= [[] for _ in range(64)] #hyperparameter needed to be changed by users
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

    return samplegaussian

class GaussianRepThread(threading.Thread):
    '''
    The class to fit gaussian distributions for each hidden node.

    parameters
    -------------------
    output: list
        A list of threads
    data: list
        A list of samples of the gaussian distribution of each hidden node
    i: int
        the index of the thread
    '''
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
    Fitting gaussian distributions for each hidden node

    parameters
    -------------------
    data: list
        A list of samples of the gaussian distribution of each hidden node
    
    return
    -------------------
    out: list
        A list of mu and sigma of each hidden node
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
def getClusterRepresentation(mus, sigmas):
    '''
    MC strategy to sample gaussian data points. Use sampled data points to fit gaussian distributions
    
    parameters
    -------------------
    mus: np.array
        mu vectors of input cells (shape: [number of cell, number of hidden nodes])
    
    sigmas: np.array
        sigma vectors of input cells (shape: [number of cell, number of hidden nodes])
   
    return
    -------------------
    fitted_gaussian_distributions: list
        A list of fitted gaussian distributions
    '''
    T1 = time.time()
    sampling = mcSampling(mus,sigmas)
    T2 = time.time()
    #print('time is %s ' %((T2-T1)))
    fitted_gaussian_distributions = fitClusterGaussianRepresentation(sampling)
    return fitted_gaussian_distributions

def normalizeDistance(distance):
    '''
    Normalize the kl divergence distance and top differential gene distances. (Use the min-max normalization method.)
    
    parameters
    -------------------
    distance: list
        A list of distance metrics. ([gaussian kl divergence, top differential gene difference])
    
    return
    -------------------

    normalizedDistance: np.array
        Sum of normalized kl divergence and top differential gene difference distance
    '''
    

    distance = np.array(distance)
    
    gaussiankld = np.array(distance[:,0])

    de = np.array(distance[:,1])
    if gaussiankld.shape[0] != 1:
        #min max normalization
        de_minmax = (de - de.min()) / (de.max() - de.min())
        ggkld = gaussiankld
        #min max normalization
        ggkld_minmax = (ggkld - ggkld.min()) / (ggkld.max() - ggkld.min())
    else:
        ggkld_minmax= gaussiankld
        de_minmax = de
    
    normalizedDistance = ggkld_minmax+de_minmax
    return normalizedDistance
def calculateKL(cluster1gaussian, cluster2gaussian):
    '''
    calculate KL divergence of multivariate gaussian distributions between two clusters.
    
    parameters
    ------------------- 

    cluster1gaussian: list
        A list of [mean, std] of gaussian distribution of cluster 1

    cluster2gaussian: list
        A list of [mean, std] of gaussian distribution of cluster 2

    
    return
    -------------------

    kl: np.array
        kl divergence of multivariate gaussian distributions between two clusters.
    '''
    cluster1 = np.array(cluster1gaussian,dtype=np.float64).reshape(-1,2)
    cluster2 = np.array(cluster2gaussian,dtype=np.float64).reshape(-1,2)
    
    std1 = cluster1[:,1]
    covariance1 = torch.tensor(np.diag(std1**2))

    mean1 = cluster1[:,0]

    p = MultivariateNormal(torch.tensor(mean1), covariance1)
    mean2 = cluster2[:,0]
    std2 = cluster2[:,1]
    covariance2 = torch.tensor(np.diag(std2**2))
    q = MultivariateNormal(torch.tensor(mean2), covariance2)
    kl = torch.distributions.kl.kl_divergence(p, q)
    
    kl = kl.detach().numpy()

    return kl





