import scanpy as sc
import os
import numpy as np
import torch
# from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms
import scanpy as sc
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
# import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
from scipy.sparse import csc_matrix
import umap
from .processTFs import mySigmoid
from scvi import data
import scvi
#NUMGENES=len(adata.var)
# customized h5ad dataset
class H5ADataSetTrainClassifier(Dataset):
    def __init__(self,fname):
        self.data=fname
        self.stagelen = len([1,2,3,4])
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):
        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        label = self.data.obs['stage'].tolist()[idx]
        label = F.one_hot(torch.tensor(label), num_classes=self.stagelen).float()
        return x_tensor,label
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
# customized h5ad dataset
class H5ADataSet(Dataset):
    def __init__(self,fname):
        self.data=fname
        # self.neighbors = neighbors
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):

        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        return x_tensor,self.data.obsp['gcn_connectivities'][idx].indices.tolist(),idx#,self.data.obs['Sample.ID'][idx]
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
class plainH5ADataSet(Dataset):
    def __init__(self,fname):
        self.data=fname
        # self.neighbors = neighbors
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):

        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        return x_tensor
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
class H5ADataSetTesting(Dataset):
    def __init__(self,fname):
        self.data=fname
        #self.data.X = self.data.layers['precompute']
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):
        x=csc_matrix(self.data.layers['precompute'][idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        return x_tensor
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
class H5ADataSetGeneWeight(Dataset):
    def __init__(self,fname):
        self.data=fname
    
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):
        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        gw = self.data.layers['geneWeight'][idx][0].toarray()[0]
        gw = mySigmoid(gw.astype(np.float32))
        gw_tensor = torch.from_numpy(gw)
        return x_tensor, gw_tensor
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
class H5ADFile:
    def __init__(self,fname):

        self.d1=fname
        
        
    def preprocess(self):
        datControl = self.d1
        #datControl=adata.raw.to_adata()
        sc.pp.filter_cells(datControl, min_genes=200)
        sc.pp.filter_genes(datControl, min_cells=3) 
        #sc.pp.highly_variable_genes(datControl, n_top_genes=5000)
        #datControl = datControl[:, datControl.var.highly_variable]
        datControl.var['mt'] = datControl.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(datControl, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        sc.pl.violin(datControl, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)
        datControl = datControl[datControl.obs.pct_counts_mt < 5, :]
        #sc.pp.normalize_total(datControl, target_sum=1e4)
        #sc.pp.log1p(datControl)
        self.d1=datControl 
        
    def cluster(self):
        datControl=self.d1
        sc.tl.pca(datControl, svd_solver='arpack')
        sc.pp.neighbors(datControl, n_neighbors=25, n_pcs=40)
        sc.tl.diffmap(datControl)
        sc.tl.leiden(datControl,resolution=0.7)
        sc.tl.paga(datControl)
        sc.pl.paga(datControl,color='leiden')
        sc.tl.umap(datControl,init_pos='paga')
        sc.pl.umap(datControl,color='leiden',legend_loc="on data")
        self.d1=datControl
        
    def getDE(self):
        datControl=self.d1
        sc.tl.rank_genes_groups(datControl, 'leiden', method='wilcoxon',n_genes=100)