'''
This is the main module of UNAGI. It contains the UNAGI class, which is the main class of UNAGI. It also contains the functions to prepare the data, start the model training and start analysing the perturbation results. Initially, `setup_data` function should be used to prepare the data. Then, `setup_training`` function should be used to setup the training parameters. Finally, `run_UNAGI` function should be used to start the model training. After the model training is done, `analyse_UNAGI` function should be used to start the perturbation analysis.
'''
import subprocess
import numpy as np
from .utils import split_dataset_into_stage, get_all_adj_adata
import os
import scanpy as sc
import gc
from .gcn_utilis import get_gcn_exp
from .runner import UNAGI_runner
import torch
from .pyro_models import VAE,Discriminator
from .UNAGI_analyst import analyst
from .trainer import UNAGI_trainer
class UNAGI:
    '''
    The UNAGI class is the main class of UNAGI. It contains the function to prepare the data, start the model training and start analysing the perturbation results.
    '''
    def __init__(self,):
        pass
    def setup_data(self, data_path,stage_key,total_stage,gcn_connectivities=False,neighbors=25,threads = 20):
        '''
        The function to specify the data directory, the attribute name of the stage information and the total number of time stages of the time-series single-cell data. If the input data is a single h5ad file, then the data will be split into multiple h5ad files based on the stage information. The function can take either the h5ad file or the directory as the input. The function will check weather the data is already splited into stages or not. If the data is already splited into stages, the data will be directly used for training. Otherwise, the data will be split into multiple h5ad files based on the stage information. The function will also calculate the cell graphs for each stage. The cell graphs will be used for the graph convolutional network (GCN) based cell graph construction.
        parameters:
        data_path: the directory of the h5ad file or the folder contains data.
        '''
        if total_stage <= 2:
            raise ValueError('The total number of stages should be larger than 2')
        
        if os.path.isfile(data_path):
            self.data_folder = os.path.dirname(data_path)
        else:
            self.data_folder = data_path
        #os.path.dirname(data_path)
        self.stage_key = stage_key
        if os.path.exists(os.path.join(self.data_folder ,'0.h5ad')):
            temp = sc.read(os.path.join(self.data_folder , '0.h5ad'))
            self.input_dim = temp.shape[1]
            if 'gcn_connectivities' not in list(temp.obsp.keys()):
                gcn_connectivities = False
            else:
                gcn_connectivities = True
        else:
            print('The dataset is not splited into stages, please use setup_data function to split the dataset into stages first')
            self.data_path = data_path
            split_dataset_into_stage(self.data_path, self.data_folder, self.stage_key)
            gcn_connectivities = False
            
        self.data_path = os.path.join(data_path,'0.h5ad')
        
        self.ns = total_stage
        #data folder is the folder that contains all the h5ad files
        self.data_folder = data_path#os.path.dirname(data_path)
        dir1 = os.path.join(self.data_folder , '0')
        dir2 = os.path.join(self.data_folder , '0/stagedata')
        dir3 = os.path.join(self.data_folder , 'model_save')
        initalcommand = 'mkdir '+ dir1 +' && mkdir '+dir2 +' && mkdir '+dir3
        p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)

        if not gcn_connectivities:
            print('Cell graphs not found, calculating cell graphs for individual stages! Using K=%d and threads=%d for cell graph construction'%(neighbors,threads))
            self.calculate_neighbor_graph(neighbors,threads)
        
        
    def calculate_neighbor_graph(self, neighbors=25,threads = 20):
        '''
        The function to calculate the cell graphs for each stage. The cell graphs will be used for the graph convolutional network (GCN) based cell graph construction.
        parameters:
        neighbors: the number of neighbors for each cell, default is 25.
        threads: the number of threads for the cell graph construction, default is 20.
        '''
        get_gcn_exp(self.data_folder, self.ns ,neighbors,threads= threads)
    def setup_training(self,
                 task, 
                 dist,
                 device=None,
                 epoch_iter=10,
                 epoch_initial=20,
                 lr=1e-4,
                 lr_dis = 5e-4,
                 beta=1,
                 hiddem_dim=256,
                 latent_dim=64,
                 graph_dim=1024,
                 BATCHSIZE=512,
                 max_iter=10,
                 GPU=False):
        '''
        Set up the training parameters and the model parameters.
        parameters:
        task: the name of this task. It is used to name the output folder.
        dist: the distribution of the single-cell data. Chosen from 'ziln' (zero-inflated log normal), 'zinb' (zero-inflated negative binomial), 'zig' (zero-inflated gamma), and 'nb' (negative binomial).
        device: the device to run the model. If GPU is enabled, the device should be specified. Default is None.
        epoch_iter: the number of epochs for the iterative training process. Default is 10.
        epoch_initial: the number of epochs for the inital iteration. Default is 20.
        lr: the learning rate of the VAE model. Default is 1e-4.
        lr_dis: the learning rate of the discriminator. Default is 5e-4.
        beta: the beta parameter of the beta-VAE. Default is 1.
        hiddem_dim: the hidden dimension of the VAE model. Default is 256.
        latent_dim: the latent dimension of the VAE model. Default is 64.
        graph_dim: the dimension of the GCN layer. Default is 1024.
        BATCHSIZE: the batch size for the model training. Default is 512.
        max_iter: the maximum number of iterations for the model training. Default is 10.
        GPU: whether to use GPU for the model training. Default is False.
        '''
        self.dist = dist
        self.device = device
        self.epoch_iter = epoch_iter
        self.epoch_initial = epoch_initial
        self.lr = lr
        self.beta = beta
        self.lr_dis = lr_dis
        self.task = task
        self.latent_dim = latent_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hiddem_dim
        self.BATCHSIZE = BATCHSIZE
        self.max_iter = max_iter
        self.GPU = GPU
        if self.GPU:
            assert self.device is not None, "GPU is enabled but device is not specified"
            self.device = torch.device(self.device)
        else:
            self.device = torch.device('cpu')

        self.model = VAE(self.input_dim, self.latent_dim, self.hidden_dim,beta=1,distribution=self.dist)
        # self.dis_model = Discriminator(self.input_dim)
        self.unagi_trainer = UNAGI_trainer(self.model,self.task,self.BATCHSIZE,self.epoch_initial,self.epoch_iter,self.device,self.lr, self.lr_dis,cuda=self.GPU)
    def run_UNAGI(self,idrem_dir):
        '''
        The function to launch the model training. The model will be trained iteratively. The number of iterations is specified by the `max_iter` parameter in the `setup_training` function.
        parameters:
        idrem_dir: the directory to the iDREM tool which is used to reconstruct the temporal dynamics.
        '''
        for iteration in range(0,self.max_iter):
            
            if iteration != 0:
                dir1 = os.path.join(self.data_folder , str(iteration))
                dir2 = os.path.join(self.data_folder , str(iteration)+'/stagedata')
                dir3 = os.path.join(self.data_folder , 'model_save')
                initalcommand = 'mkdir '+ dir1 +' && mkdir '+dir2
                p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
            unagi_runner = UNAGI_runner(self.data_folder,self.ns,iteration,self.unagi_trainer,idrem_dir)
            unagi_runner.run()

  
    def analyse_UNAGI(self,data_path,iteration,progressionmarker_background_sampling_times,target_dir=None,customized_drug=None,cmap_dir=None):
        '''
        Perform downstream tasks including dynamic markers discoveries, hierarchical markers discoveries, pathway perturbations and compound perturbations.
        parameters:
        data_path: the directory of the data (h5ad format, e.g. org_dataset.h5ad).
        iteration: the iteration used for analysis.
        progressionmarker_background_sampling_times: the number of times to sample the background cells for dynamic markers discoveries.
        target_dir: the directory to save the results. Default is None.
        customized_drug: the customized drug perturbation list. Default is None.
        cmap_dir: the directory to the cmap database. Default is None.
        '''
        analysts = analyst(data_path,iteration,target_dir=target_dir,customized_drug=customized_drug,cmap_dir=cmap_dir)
        analysts.start_analyse(progressionmarker_background_sampling_times)
        print('The analysis has been done, please check the outputs!')


        

        


        

