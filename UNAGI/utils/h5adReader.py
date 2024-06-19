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
from ..dynamic_regulatory_networks.processTFs import mySigmoid
#NUMGENES=len(adata.var)
# customized h5ad dataset
class H5ADDataSet(Dataset):
    '''
    The customized dataset for the data without gene weights. (For the initial iteration) The dataset will return the gene expression and the cell graph for each cell.
    '''
    def __init__(self,fname):
        self.data=fname
        # self.neighbors = neighbors
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):

        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        if 'gcn_connectivities' not in self.data.obsp.keys():
            return x_tensor,None,idx
        return x_tensor,self.data.obsp['gcn_connectivities'][idx].indices.tolist(),idx#,self.data.obs['Sample.ID'][idx]
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
class H5ADPlainDataSet(Dataset):
    '''
    The customized dataset for the data without gene weights. (For the initial iteration) The dataset will return the gene expression and the cell graph for each cell.
    '''
    def __init__(self,fname):
        self.data=fname
        # self.neighbors = neighbors
    def __len__(self):
        return self.data.X.shape[0]
    
    def __getitem__(self,idx):

        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        if 'gcn_connectivities' not in self.data.obsp.keys():
            return x_tensor,0,idx
        return x_tensor,0,idx#,self.data.obs['Sample.ID'][idx]
    def num_genes(self):
        return len(self.data.var)
    def returnadata(self):
        return self.data
    
class H5ADataSetGeneWeight(Dataset):
    '''
    The customized dataset for the data with gene weights. The dataset will return the gene expression and the gene weight for each cell.
    '''
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
