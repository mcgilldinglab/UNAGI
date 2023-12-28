import subprocess
import numpy as np
from .utils import split_dataset_into_stage, get_all_adj_adata
import os
import scanpy as sc
import gc
from .gcn_utilis import get_gcn_exp
from .UNAGI_runner import UNAGI_runner
import torch
from .pyro_models import VAE,Discriminator
from .UNAGI_analyst import analyst
from .trainer import UNAGI_trainer
class UNAGI():
    def __init__(self):
        super(UNAGI, self).__init__()
    def setup_data(self, data_path,stage_key,total_stage,gcn_connectivities=False,neighbors=25,threads = 20):
        #if the input data is a single h5ad file, then get the directory of the file
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
            self.split_dataset_into_stage()
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
        
    def split_dataset_into_stage(self):
        split_dataset_into_stage(self.data_path, self.data_folder, self.stage_key)
    def calculate_neighbor_graph(self, neighbors=25,threads = 20):
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
        analysts = analyst(data_path,iteration,target_dir=target_dir,customized_drug=customized_drug,cmap_dir=cmap_dir)
        analysts.start_analyse(progressionmarker_background_sampling_times)
        print('The analysis has been done, please check the outputs!')


        

        


        

