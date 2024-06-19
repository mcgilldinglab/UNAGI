import scanpy as sc
import os
import gc
import torch
from torch.utils.data import DataLoader
from ..utils.gcn_utils import setup_graph
import pyro
import numpy as np
from ..utils.trainer_utils import transfer_to_ranking_score
import scipy.sparse as sp
from ..utils.h5adReader import H5ADataSet
#import variable
from ..train.customized_elbo import *
from pyro.optim import Adam
class UNAGI_trainer():
    '''
    The trainer class is the class to train the UNAGI model. The trainer class will train the model for a given number of epochs. 
    
    parameters
    ----------------
    model: object
        the model to be trained.
    modelName: str
        the name of the model.
    batch_size: int
        the batch size for training.
    epoch_initial: int
        the initial epoch for training.
    epoch_iter: int
        the total number of epochs for training.
    device: str
        the device to train the model.
    lr: float
        the learning rate for the variational autoencoder (VAE) model.
    lr_dis: float
        the learning rate for the discriminator.
    cuda: bool
        whether to use GPU for the model training. Default is True.
    

    '''
    def __init__(self,model, modelName,batch_size,epoch_initial,epoch_iter,device,lr, lr_dis,cuda=True):
        super(UNAGI_trainer, self).__init__()
        self.model = model
        self.modelName = modelName
        self.epoch_initial = epoch_initial
        self.epoch_iter = epoch_iter
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = device
        self.lr = lr
        self.lr_dis = lr_dis
    def train_model(self,adata, vae, train_loader,adj, adversarial = True,geneWeights=None, use_cuda=True):
        '''
        The function to train the model.

        parameters:
        ----------------

        adata: anndata
            the single-cell data.
        vae: model
            the model to be trained.
        train_loader: DataLoader
            the data loader for the model training.
        adj: scipy.sparse.csr_matrix
            the adjacency matrix of cell graphs.
        geneWeights: np.array
            the gene weights for the model training. Default is None.
        use_cuda: bool
            whether to use GPU for the model training. Default is True.

        return:
        ----------------
        total_epoch_loss_train: float
            the total loss for the model training.
        '''
        # initialize loss accumulator
        epoch_loss = 0.
        
        if use_cuda:
            adj=adj.to(self.device)
        
        placeholder = torch.zeros(adata.X.shape,dtype=torch.float32)
        optimizer_vae = Adam({"lr": self.lr})
            
        optimizer_dis = Adam({"lr": self.lr_dis}) 

    # do a training epoch over each mini-batch x
        epoch_loss = 0.
        loss1 = 0
        loss2 = 0
        for i, [x,neighbourhoods,idx] in enumerate(train_loader): 
            temp_x = placeholder.clone()
            start = i*self.batch_size
            if (1+i)*self.batch_size > len(adj):
                end =  len(adj)
            else:
                end = (1+i)*self.batch_size
            # neighbourhood = neighbourhoods[i]
            neighbourhood = [item for sublist in neighbourhoods for item in sublist]
            # neighbourhood = list(set([item for sublist in neighbourhoods for item in sublist]))
            temp_x[neighbourhood] = torch.Tensor(adata.X)[neighbourhood]
            x = temp_x

        # if on GPU put mini-batch into CUDA memory
            if self.cuda:
                x = x.to(self.device)
            if geneWeights is not None:
                geneWeights1 = torch.tensor(transfer_to_ranking_score(geneWeights[idx].toarray()))
                geneWeights1 = geneWeights1.to(self.device)
                # if adversarial:
                loss = graphUpdater(myELBO(geneWeights1),vae.model,vae.guide,vae.discriminator,optimizer_vae,x,adj,i, start,end,device=self.device, second_optimizer=optimizer_dis)
                # else:
                    # loss = myELBO(geneWeights1)(vae.model,vae.guide,x,adj,i, start,end,device=self.device)
            else:
                # if adversarial:
                loss = graphUpdater(myELBO(),vae.model,vae.guide,vae.discriminator,optimizer_vae,x,adj,i, start,end,device=self.device,second_optimizer=optimizer_dis)
            
            # loss = graphUpdater(myELBO(),vae.model,vae.guide,vae.discriminator,optimizer_vae,x,adj,i, start,end,self.device=self.device,second_optimizer=optimizer_dis)
            epoch_loss += loss
            

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        # print('loss1', loss1/normalizer_train)
        # print('loss2', loss2/normalizer_train)
        print('loss', total_epoch_loss_train)
        return total_epoch_loss_train
    def get_latent_representation(self,adata,iteration,target_dir):
        '''
        Retrieve the latent representation of the single-cell data.

        parameters:
        ----------------
        adata: anndata
            the single-cell data.   
        iteration: int
            the iteration used for analysis.
        target_dir: str
            the directory of the task.

        return:
        ----------------
        z_locs: np.array
            the mean of the latent representation.
        z_scales: np.array
            the variance of the latent representation.
        TZ: np.array
            the latent representation.
        '''

        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
    
        if 'gcn_connectivities' in adata.obsp.keys():
            adj = adata.obsp['gcn_connectivities']
            adj = adj.asformat('coo')
        cell = H5ADataSet(adata)
        num_genes=cell.num_genes()
        placeholder = torch.zeros(adata.X.shape,dtype=torch.float32)
        cell_loader=DataLoader(cell,batch_size=self.batch_size,num_workers=0,shuffle=False)
        self.model.load_state_dict(torch.load(os.path.join(target_dir,'model_save/'+self.modelName+'_'+str(iteration)+'.pth'),map_location=self.device))
        TZ=[]
        z_locs = []
        z_scales = []
        adj = setup_graph(adj)
        adj = adj.to(self.device)
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        for i, [x,neighbourhoods,idx] in enumerate(cell_loader):
            temp_x = placeholder.clone()
            start = i*self.batch_size
            if (1+i)*self.batch_size > len(adj):
                end =  len(adj)
            else:
                end = (1+i)*self.batch_size
            neighbourhood = [item for sublist in neighbourhoods for item in sublist]
            temp_x[neighbourhood] = torch.Tensor(adata.X)[neighbourhood]
            x = temp_x
            if self.cuda:
                x = x.to(self.device)
            _,mu, logvar,_,_  = self.model.getZ(x.view(-1, num_genes),adj,i,start, end,test=False)
            # mu, logvar = self.model.encoder(x.view(-1, num_genes),adj,idx)
            z = mu+logvar
            z_locs+=mu.detach().cpu().numpy().tolist()
            z_scales+=logvar.detach().cpu().numpy().tolist()
            TZ+=z.detach().cpu().numpy().tolist()
        z_locs = np.array(z_locs)
        z_scales = np.array(z_scales)
        TZ = np.array(TZ)
        return z_locs, z_scales, TZ

    def train(self, adata, iteration, target_dir, is_iterative=False):
        '''
        The function to train the model.

        parameters
        ----------------
        adata: anndata
            the single-cell data.
        iteration: int
            the iteration used for analysis.
        target_dir: str 
            the directory of the task.
        is_iterative: bool
            whether to use the iterative training strategy. Default is False.
        '''
        
        assert 'X_pca' in adata.obsm.keys(), 'PCA is not performed'
        if 'X_pca' not in adata.obsm.keys():
            sc.tl.pca(adata, svd_solver='arpack')

        if 'gcn_connectivities' in adata.obsp.keys():
            adj = adata.obsp['gcn_connectivities']
            adj = adj.asformat('coo')
        adj = setup_graph(adj)
        if is_iterative:
            geneWeights = adata.layers['geneWeight']
            cell = H5ADataSet(adata)
        else:
            geneWeights = None
            cell = H5ADataSet(adata)
            
        cell_loader = DataLoader(cell, batch_size=self.batch_size, num_workers=0, shuffle=True)
        
        pyro.clear_param_store()
        print('...')  

        if os.path.exists(os.path.join(target_dir, 'model_save', self.modelName + '_' + str(iteration-1) + '.pth')):
            vae = self.model
            # dis = self.discriminator
            if os.path.exists(os.path.join(target_dir, 'model_save', self.modelName + '_' + str(iteration) + '.pth')):
                print('load current iteration model....')
                # dis.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth')))
                vae.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth')))
            else:
                print('load last iteration model.....')
            
                # dis.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration-1) + '.pth')))
                vae.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration-1) + '.pth')))
        else:
            vae = self.model

        vae.to(self.device)
        if geneWeights is None and is_iterative:
            print('no geneWeight')
    
        gc.collect()

        train_elbo = []
        epoch_range = self.epoch_iter if is_iterative else self.epoch_initial
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        for epoch in range(epoch_range):
            print(epoch)
            total_epoch_loss_train = self.train_model(adata, vae, cell_loader, adj, geneWeights=geneWeights if is_iterative else None, use_cuda=self.cuda)
            train_elbo.append(-total_epoch_loss_train)
            print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
            with open(os.path.join(target_dir, '%d/loss.txt' % (int(iteration))), "a+") as f:
                f.write("[epoch %03d]  average training loss: %.4f\n" % (epoch, total_epoch_loss_train))
                f.close()
        torch.save(vae.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth'))
        # torch.save(dis, os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth'))
        # torch.save(vae.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth'))
        # torch.save(dis.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth'))
    