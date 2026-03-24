import scanpy as sc
import os
import gc
import torch
from torch.utils.data import DataLoader
from ..utils.gcn_utils import setup_graph
import pyro
from pathlib import Path
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
        self._loaded_checkpoint = None

    def _ensure_dense_expression(self, adata):
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.toarray()
        elif not isinstance(adata.X, np.ndarray):
            adata.X = np.asarray(adata.X)
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32, copy=False)
        return adata.X

    def _expression_tensor(self, adata):
        return torch.from_numpy(self._ensure_dense_expression(adata))

    def _graph_csr(self, adata):
        if 'gcn_connectivities' not in adata.obsp.keys():
            return None
        return adata.obsp['gcn_connectivities'].tocsr()

    def _batch_neighbourhood_nodes(self, graph_csr, batch_idx):
        batch_np = batch_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        indptr = graph_csr.indptr
        indices = graph_csr.indices
        chunks = [batch_np]
        for each in batch_np:
            chunks.append(indices[indptr[each]:indptr[each + 1]])
        nodes = np.unique(np.concatenate(chunks))
        return torch.as_tensor(nodes, dtype=torch.long)

    def _prepare_plain_batch(self, x_source, batch_idx):
        x = x_source.index_select(0, batch_idx.cpu())
        if self.cuda:
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
        return x, batch_idx

    def _prepare_gcn_batch(self, x_source, graph_csr, work_x, batch_idx):
        touched_nodes = self._batch_neighbourhood_nodes(graph_csr, batch_idx)
        source_rows = x_source.index_select(0, touched_nodes)
        if self.cuda:
            touched_nodes = touched_nodes.to(self.device)
            source_rows = source_rows.to(self.device)
            batch_idx = batch_idx.to(self.device)
        work_x.index_copy_(0, touched_nodes, source_rows)
        return work_x, batch_idx, touched_nodes

    def _ensure_loaded_checkpoint(self, iteration, target_dir):
        target_dir = Path(target_dir)
        checkpoint_path = target_dir / 'model_save' / (self.modelName + '_' + str(iteration) + '.pth')
        cache_key = (str(checkpoint_path), str(self.device))
        if self._loaded_checkpoint == cache_key:
            return
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self._loaded_checkpoint = cache_key
    def train_model(self,adata, vae,dis, train_loader,adj, x_source, graph_csr=None, adversarial=True,geneWeights=None, use_cuda=True):
        # initialize loss accumulator
        epoch_loss = 0.
        criterion=nn.BCELoss().to(self.device)
        if use_cuda:
            if adj is not None:
                adj=adj.to(self.device)

        work_x = None
        if self.GCN:
            buffer_device = self.device if self.cuda else torch.device('cpu')
            work_x = torch.zeros(adata.X.shape, dtype=torch.float32, device=buffer_device)
        optimizer_vae = optim.Adam(lr= self.lr,params=vae.parameters())
        if adversarial:
            optimizer_dis = optim.Adam(lr=self.lr_dis,params=dis.parameters()) 

    # do a training epoch over each mini-batch x
        vae_loss = 0
        dis_loss = 0
        adversarial_loss = 0
        
        for i, idx in enumerate(train_loader): 
            idx = idx.long()
            size = len(idx)
            if self.GCN:
                x, idx_device, touched_nodes = self._prepare_gcn_batch(x_source, graph_csr, work_x, idx)
            else:
                x, idx_device = self._prepare_plain_batch(x_source, idx)
                touched_nodes = None
            if geneWeights is not None:
                idx_np = idx.detach().cpu().numpy()
                geneWeights1 = torch.tensor(transfer_to_ranking_score(geneWeights[idx_np].toarray())).to(self.device)
                if self.GCN:
                    mu, dropout_logits, mu_, logvar_,_ = vae(x,adj,idx_device)
                    loss =  vae.loss_function(x[idx_device,:], mu, dropout_logits, mu_, logvar_,gene_weights=geneWeights1)
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
                if self.GCN:
                    mu, dropout_logits, mu_, logvar_,_ = vae(x,adj,idx_device)
                    loss =  vae.loss_function(x[idx_device,:], mu, dropout_logits, mu_, logvar_)
                else:
                    mu, dropout_logits, mu_, logvar_,recons = vae(x) 
                    loss =  vae.loss_function(x, mu, dropout_logits, mu_, logvar_)
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()
                vae_loss += loss.item()
            if adversarial:

                #discriminator loss
                if self.GCN:
                    _,_,_,_,recons = vae(x,adj,idx_device)
                else:
                    _,_,_,_,recons = vae(x)
                zeros_label1=Variable(torch.zeros(size,1)).to(self.device)
                ones_label = torch.ones((size,1)).to(self.device)
                zeros_label = torch.zeros((size,1)).to(self.device)
                if self.GCN:
                    output_real = dis(x[idx_device,:])
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
                if self.GCN:
                    _,_,_,_,recons = vae(x,adj,idx_device)
                else:
                    _,_,_,_,recons = vae(x)
                output_fake = dis(recons)
                loss_adversarial = criterion(output_fake,ones_label)
                optimizer_vae.zero_grad()
                loss_adversarial.backward()
                optimizer_vae.step()
                adversarial_loss += loss_adversarial.item()
            if touched_nodes is not None:
                work_x.index_fill_(0, touched_nodes, 0.0)

            

        normalizer_train = len(train_loader)
        total_epoch_vae_loss = vae_loss / normalizer_train
        if adversarial:
            total_epoch_dis_loss = dis_loss / normalizer_train
            total_epoch_adversarial_loss = adversarial_loss / normalizer_train
        return total_epoch_vae_loss
    def get_latent_representation(self,adata,iteration,target_dir):
        '''
        find out the best groups of resolution for clustering
        '''
        target_dir = Path(target_dir)
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
    
        graph_csr = self._graph_csr(adata)
        if self.GCN:
            cell = H5ADDataSet(adata)
        else:
            cell = H5ADPlainDataSet(adata)
        num_genes=cell.num_genes()
        cell_loader=DataLoader(cell,batch_size=self.batch_size,num_workers=0)
        self._ensure_loaded_checkpoint(iteration, target_dir)
        TZ=[]
        z_locs = []
        z_scales = []
        x_source = self._expression_tensor(adata)
        adj = None
        work_x = None
        if self.GCN and graph_csr is not None:
            adj = setup_graph(graph_csr.asformat('coo')).to(self.device)
            buffer_device = self.device if self.cuda else torch.device('cpu')
            work_x = torch.zeros(adata.X.shape, dtype=torch.float32, device=buffer_device)
        for i, idx in enumerate(cell_loader):
            idx = idx.long()
            if self.GCN:
                x, idx_device, touched_nodes = self._prepare_gcn_batch(x_source, graph_csr, work_x, idx)
                mu, logvar = self.model.encoder(x.view(-1, num_genes),adj,idx_device)
            else:
                x, _ = self._prepare_plain_batch(x_source, idx)
                mu, logvar = self.model.encoder(x.view(-1, num_genes))
            z = mu+logvar
            z_locs+=mu.detach().cpu().numpy().tolist()
            z_scales+=logvar.detach().cpu().numpy().tolist()
            TZ+=z.detach().cpu().numpy().tolist()
            if self.GCN:
                work_x.index_fill_(0, touched_nodes, 0.0)
        z_locs = np.array(z_locs)
        z_scales = np.array(z_scales)
        z_scales = np.exp(0.5 * z_scales)
        TZ = np.array(TZ)
        return z_locs, z_scales, TZ
    
    def get_reconstruction(self,adata,iteration,target_dir):
        '''
        retrieve the reconstructed data
        '''
        target_dir = Path(target_dir)
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata)
    
        graph_csr = self._graph_csr(adata)
        cell = H5ADDataSet(adata)
        num_genes=cell.num_genes()
        cell_loader=DataLoader(cell,batch_size=self.batch_size,num_workers=0)
        self._ensure_loaded_checkpoint(iteration, target_dir)
        recons = []
        x_source = self._expression_tensor(adata)
        adj = None
        work_x = None
        if self.GCN and graph_csr is not None:
            adj = setup_graph(graph_csr.asformat('coo')).to(self.device)
            buffer_device = self.device if self.cuda else torch.device('cpu')
            work_x = torch.zeros(adata.X.shape, dtype=torch.float32, device=buffer_device)
        for i, idx in enumerate(cell_loader):
            idx = idx.long()
            if self.GCN:
                x, idx_device, touched_nodes = self._prepare_gcn_batch(x_source, graph_csr, work_x, idx)
                _, _, _, _, recon = self.model(x.view(-1, num_genes),adj,idx_device)
            else:
                x, _ = self._prepare_plain_batch(x_source, idx)
                _, _, _, _, recon = self.model(x.view(-1, num_genes))
            
            recons+=recon.detach().cpu().numpy().tolist()
            if self.GCN:
                work_x.index_fill_(0, touched_nodes, 0.0)
           
        recons = np.array(recons)

        return recons
    def train(self, adata, iteration, target_dir, adversarial=True,is_iterative=False):
        target_dir = Path(target_dir)
        assert 'X_pca' in adata.obsm.keys(), 'PCA is not performed'
        if 'X_pca' not in adata.obsm.keys():
            sc.tl.pca(adata, svd_solver='arpack')

        graph_csr = self._graph_csr(adata)
        adj = setup_graph(graph_csr.asformat('coo')) if graph_csr is not None else None
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

        if os.path.exists(target_dir / 'model_save' / (self.modelName + '_' + str(iteration-1) + '.pth')):
            vae = self.model
            dis = self.dis_model
            # dis = self.discriminator
            if os.path.exists(target_dir / 'model_save' / (self.modelName + '_' + str(iteration) + '.pth')):
                print(f"[TRAIN] iteration {iteration}: resume from existing checkpoint")
                if adversarial:
                    dis.load_state_dict(torch.load(target_dir / 'model_save' / (self.modelName + '_dis_' + str(iteration) + '.pth')))
                vae.load_state_dict(torch.load(target_dir / 'model_save' / (self.modelName + '_' + str(iteration) + '.pth')))
            else:
                print(f"[TRAIN] iteration {iteration}: initialize from previous iteration checkpoint")
                if adversarial:
                    dis.load_state_dict(torch.load(target_dir / 'model_save' / (self.modelName + '_dis_' + str(iteration-1) + '.pth')))
                vae.load_state_dict(torch.load(target_dir / 'model_save' / (self.modelName + '_' + str(iteration-1) + '.pth')))
        else:
            vae = self.model
            dis = self.dis_model

        vae.to(self.device)
        if adversarial:
            dis.to(self.device)

        gc.collect()

        train_elbo = []
        epoch_range = self.epoch_iter if is_iterative else self.epoch_initial
        x_source = self._expression_tensor(adata)
        for epoch in range(epoch_range):
            total_epoch_loss_train = self.train_model(
                adata,
                vae,
                dis,
                cell_loader,
                adj,
                x_source,
                graph_csr=graph_csr,
                adversarial=adversarial,
                geneWeights=geneWeights if is_iterative else None,
                use_cuda=self.cuda,
            )
            train_elbo.append(-total_epoch_loss_train)
            print(
                f"[UNAGI-TRAIN] iteration={iteration} epoch={epoch + 1}/{epoch_range} "
                f"loss={total_epoch_loss_train:.4f}"
            )
            print(f"[TRAIN] iteration {iteration} epoch {epoch + 1}/{epoch_range} loss={total_epoch_loss_train:.4f}")
            with open(target_dir / f'{int(iteration)}' / 'loss.txt', "a+") as f:
                f.write("[epoch %03d]  average training loss: %.4f\n" % (epoch, total_epoch_loss_train))
                f.close()
        torch.save(vae.state_dict(), target_dir / 'model_save' / (self.modelName + '_' + str(iteration) + '.pth'))
        # torch.save(dis, target_dir / 'model_save' / (self.modelName + '_dis_' + str(iteration) + '.pth'))
        # torch.save(vae.state_dict(), target_dir / 'model_save' / (self.modelName + '_' + str(iteration) + '.pth'))
        if adversarial:
            torch.save(dis.state_dict(), target_dir / 'model_save' / (self.modelName + '_dis_' + str(iteration) + '.pth'))
