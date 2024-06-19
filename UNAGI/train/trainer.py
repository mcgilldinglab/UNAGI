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
from ..utils.h5adReader import H5ADDataSet,H5ADPlainDataSet
#import variable
from ..train.customized_elbo import *
import torch.nn as nn
from pyro.optim import Adam
from torch import optim
#import variable
from torch.autograd import Variable

class UNAGI_trainer():
    def __init__(self,model, dis_model,modelName,batch_size,epoch_initial,epoch_iter,device,lr, lr_dis,GCN=True,cuda=True):
        super(UNAGI_trainer, self).__init__()
        self.model = model
        self.modelName = modelName
        self.epoch_initial = epoch_initial
        self.epoch_iter = epoch_iter
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = device
        self.lr = lr
        self.dis_model = dis_model
        self.GCN = GCN
        self.lr_dis = lr_dis
    def train_model(self,adata, vae,dis, train_loader,adj, adversarial=True,geneWeights=None, use_cuda=True):
        # initialize loss accumulator
        epoch_loss = 0.
        criterion=nn.BCELoss().to(self.device)
        if use_cuda:
            if adj is not None:
                adj=adj.to(self.device)

        
        placeholder = torch.zeros(adata.X.shape,dtype=torch.float32)
        optimizer_vae = optim.Adam(lr= self.lr,params=vae.parameters())
        if adversarial:
            optimizer_dis = optim.Adam(lr=self.lr_dis,params=dis.parameters()) 

    # do a training epoch over each mini-batch x
        vae_loss = 0
        dis_loss = 0
        adversarial_loss = 0
        
        for i, [x,neighbourhoods,idx] in enumerate(train_loader): 
            size = len(x)
            if self.GCN:
                
                temp_x = placeholder.clone()
                start = i*self.batch_size
                if (1+i)*self.batch_size > len(adj):
                    end =  len(adj)
                else:
                    end = (1+i)*self.batch_size
                neighbourhood = [item for sublist in neighbourhoods for item in sublist]
                temp_x[neighbourhood] = torch.Tensor(adata.X)[neighbourhood]
                x = temp_x
            else:
                neighbourhood = None
        # if on GPU put mini-batch into CUDA memory
            if self.cuda:
                x = x.to(self.device)
            if geneWeights is not None:
                geneWeights1 = torch.tensor(transfer_to_ranking_score(geneWeights[idx].toarray()))
                geneWeights1 = geneWeights1.to(self.device)
                if neighbourhood is not None:
                    mu, dropout_logits, mu_, logvar_,_ = vae(x,adj,idx)
                    loss =  vae.loss_function(x[idx,:], mu, dropout_logits, mu_, logvar_,gene_weights=geneWeights1)
                else:
                    mu, dropout_logits, mu_, logvar_,_ = vae(x)
                    loss =  vae.loss_function(x, mu, dropout_logits, mu_, logvar_,gene_weights=geneWeights1)
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()
                vae_loss += loss.item()

                # continue
            else:
                #train the generator
                if neighbourhood is not None:
                    mu, dropout_logits, mu_, logvar_,_ = vae(x,adj,idx)
                    loss =  vae.loss_function(x[idx,:], mu, dropout_logits, mu_, logvar_)
                else:
                    mu, dropout_logits, mu_, logvar_,recons = vae(x) 
                    loss =  vae.loss_function(x, mu, dropout_logits, mu_, logvar_)
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()
                vae_loss += loss.item()
            if adversarial:

                #discriminator loss
                if neighbourhood is not None:
                    _,_,_,_,recons = vae(x,adj,idx)
                else:
                    _,_,_,_,recons = vae(x)
                zeros_label1=Variable(torch.zeros(size,1)).to(self.device)
                ones_label = torch.ones((size,1)).to(self.device)
                zeros_label = torch.zeros((size,1)).to(self.device)
                if neighbourhood is not None:
                    output_real = dis(x[idx,:])
                else:
                    output_real = dis(x)
                output_fake = dis(recons)
                loss_real = criterion(output_real,ones_label)
                loss_fake = criterion(output_fake,zeros_label)
                loss_dis = loss_real + loss_fake
                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()
                dis_loss += loss_dis.item()
                
                #tune the generator
                if neighbourhood is not None:
                    _,_,_,_,recons = vae(x,adj,idx)
                else:
                    _,_,_,_,recons = vae(x)
                output_fake = dis(recons)
                loss_adversarial = criterion(output_fake,ones_label)
                optimizer_vae.zero_grad()
                loss_adversarial.backward()
                optimizer_vae.step()
                adversarial_loss += loss_adversarial.item()

            

        normalizer_train = len(train_loader)
        total_epoch_vae_loss = vae_loss / normalizer_train
        print('vae_loss', total_epoch_vae_loss)
        if adversarial:
            total_epoch_dis_loss = dis_loss / normalizer_train
            total_epoch_adversarial_loss = adversarial_loss / normalizer_train
            print('dis_loss', total_epoch_dis_loss)
            print('adversarial_loss', total_epoch_adversarial_loss)
        return total_epoch_vae_loss
    def get_latent_representation(self,adata,iteration,target_dir):
        '''
        find out the best groups of resolution for clustering
        '''
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
    
        if 'gcn_connectivities' in adata.obsp.keys():
            adj = adata.obsp['gcn_connectivities']
            adj = adj.asformat('coo')
        if self.GCN:
            cell = H5ADDataSet(adata)
        else:
            cell = H5ADPlainDataSet(adata)
        num_genes=cell.num_genes()
        placeholder = torch.zeros(adata.X.shape,dtype=torch.float32)
        cell_loader=DataLoader(cell,batch_size=self.batch_size,num_workers=0)
        self.model.load_state_dict(torch.load(os.path.join(target_dir,'model_save/'+self.modelName+'_'+str(iteration)+'.pth'),map_location=self.device))
        TZ=[]
        z_locs = []
        z_scales = []
        adj = setup_graph(adj)
        adj = adj.to(self.device)
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        for i, [x,neighbourhoods,idx] in enumerate(cell_loader):
            if self.GCN:
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
                # _,mu, logvar,_,_  = self.model.getZ(x.view(-1, num_genes),adj,i,start, end,test=False)
                mu, logvar = self.model.encoder(x.view(-1, num_genes),adj,idx)
            else:
                if self.cuda:
                    x = x.to(self.device)
                mu, logvar = self.model.encoder(x.view(-1, num_genes))
            z = mu+logvar
            z_locs+=mu.detach().cpu().numpy().tolist()
            z_scales+=logvar.detach().cpu().numpy().tolist()
            TZ+=z.detach().cpu().numpy().tolist()
        z_locs = np.array(z_locs)
        z_scales = np.array(z_scales)
        z_scales = np.exp(0.5 * z_scales)
        TZ = np.array(TZ)
        return z_locs, z_scales, TZ
    
    def get_reconstruction(self,adata,iteration,target_dir):
        '''
        retrieve the reconstructed data
        '''
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
    
        if 'gcn_connectivities' in adata.obsp.keys():
            adj = adata.obsp['gcn_connectivities']
            adj = adj.asformat('coo')
        cell = H5ADDataSet(adata)
        num_genes=cell.num_genes()
        placeholder = torch.zeros(adata.X.shape,dtype=torch.float32)
        cell_loader=DataLoader(cell,batch_size=self.batch_size,num_workers=0)
        self.model.load_state_dict(torch.load(os.path.join(target_dir,'model_save/'+self.modelName+'_'+str(iteration)+'.pth'),map_location=self.device))
        self.model = self.model.to(self.device)
        recons = []
        adj = setup_graph(adj)
        adj = adj.to(self.device)
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        for i, [x,neighbourhoods,idx] in enumerate(cell_loader):
            if self.GCN:
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
                # _,mu, logvar,_,_  = self.model.getZ(x.view(-1, num_genes),adj,i,start, end,test=False)
                _, _, _, _, recon = self.model(x.view(-1, num_genes),adj,idx)
            else:
                if self.cuda:
                    x = x.to(self.device)
                _, _, _, _, recon = self.model(x.view(-1, num_genes))
            
            recons+=recon.detach().cpu().numpy().tolist()
           
        recons = np.array(recons)

        return recons
    def train(self, adata, iteration, target_dir, adversarial=True,is_iterative=False):
        
        assert 'X_pca' in adata.obsm.keys(), 'PCA is not performed'
        if 'X_pca' not in adata.obsm.keys():
            sc.tl.pca(adata, svd_solver='arpack')

        if 'gcn_connectivities' in adata.obsp.keys():
            adj = adata.obsp['gcn_connectivities']
            adj = adj.asformat('coo')
            adj = setup_graph(adj)
        else:
            adj = None
        if is_iterative:
            geneWeights = adata.layers['geneWeight']
        else:
            geneWeights = None
        if self.GCN:
            cell = H5ADDataSet(adata)
        else:
            cell = H5ADPlainDataSet(adata)
            
        cell_loader = DataLoader(cell, batch_size=self.batch_size, num_workers=0, shuffle=True)
        
        pyro.clear_param_store()
        print('...')  

        if os.path.exists(os.path.join(target_dir, 'model_save', self.modelName + '_' + str(iteration-1) + '.pth')):
            vae = self.model
            dis = self.dis_model
            # dis = self.discriminator
            if os.path.exists(os.path.join(target_dir, 'model_save', self.modelName + '_' + str(iteration) + '.pth')):
                print('load current iteration model....')
                if adversarial:
                    dis.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth')))
                vae.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth')))
            else:
                print('load last iteration model.....')
                if adversarial:
                    dis.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration-1) + '.pth')))
                vae.load_state_dict(torch.load(os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration-1) + '.pth')))
        else:
            vae = self.model
            dis = self.dis_model

        vae.to(self.device)
        if adversarial:
            dis.to(self.device)
        if geneWeights is None and is_iterative:
            print('no geneWeight')
    
        gc.collect()

        train_elbo = []
        epoch_range = self.epoch_iter if is_iterative else self.epoch_initial
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        for epoch in range(epoch_range):
            print(epoch)
            total_epoch_loss_train = self.train_model(adata, vae,dis, cell_loader, adj, adversarial=adversarial,geneWeights=geneWeights if is_iterative else None, use_cuda=self.cuda)
            train_elbo.append(-total_epoch_loss_train)
            print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
            with open(os.path.join(target_dir, '%d/loss.txt' % (int(iteration))), "a+") as f:
                f.write("[epoch %03d]  average training loss: %.4f\n" % (epoch, total_epoch_loss_train))
                f.close()
        torch.save(vae.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth'))
        # torch.save(dis, os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth'))
        # torch.save(vae.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_' + str(iteration) + '.pth'))
        if adversarial:
            torch.save(dis.state_dict(), os.path.join(target_dir, 'model_save/' + self.modelName + '_dis_' + str(iteration) + '.pth'))
    