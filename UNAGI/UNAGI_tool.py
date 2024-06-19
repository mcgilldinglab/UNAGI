'''
This is the main module of UNAGI. It contains the UNAGI class, which is the main class of UNAGI. It also contains the functions to prepare the data, start the model training and start analysing the perturbation results. Initially, `setup_data` function should be used to prepare the data. Then, `setup_training`` function should be used to setup the training parameters. Finally, `run_UNAGI` function should be used to start the model training. After the model training is done, `analyse_UNAGI` function should be used to start the perturbation analysis.
'''
import subprocess
from tracemalloc import start
import numpy as np
from .utils.attribute_utils import split_dataset_into_stage, get_all_adj_adata
import os
import scanpy as sc
import gc
from .utils.gcn_utils import get_gcn_exp
from .train.runner import UNAGI_runner
import torch
from .model.models import VAE,Discriminator,Plain_VAE
from .UNAGI_analyst import analyst
from .train.trainer import UNAGI_trainer
class UNAGI:
    '''
    The UNAGI class is the main class of UNAGI. It contains the function to prepare the data, start the model training and start analysing the perturbation results.
    '''
    def __init__(self,):
        self.CPO_parameters = None
        self.iDREM_parameters = None
        self.species = 'Human'
        self.input_dim = None
    def setup_data(self, data_path,stage_key,total_stage,gcn_connectivities=False,neighbors=25,threads = 20):
        '''
        The function to specify the data directory, the attribute name of the stage information and the total number of time stages of the time-series single-cell data. If the input data is a single h5ad file, then the data will be split into multiple h5ad files based on the stage information. The function can take either the h5ad file or the directory as the input. The function will check weather the data is already splited into stages or not. If the data is already splited into stages, the data will be directly used for training. Otherwise, the data will be split into multiple h5ad files based on the stage information. The function will also calculate the cell graphs for each stage. The cell graphs will be used for the graph convolutional network (GCN) based cell graph construction.
        
        parameters
        --------------
        data_path: str 
            the directory of the h5ad file or the folder contains data.
        stage_key: str
            the attribute name of the stage information.
        total_stage: int
            the total number of time stages of the time-series single-cell data.
        gcn_connectivities: bool
            whether the cell graphs are already calculated. Default is False.
        neighbors: int
            the number of neighbors for each cell used to construct the cell neighbors graph, default is 25.
        threads: int
            the number of threads for the cell graph construction, default is 20.
        '''
        if total_stage < 2:
            raise ValueError('The total number of stages should be larger than 1')
        
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
        
        parameters
        --------------
        neighbors: int
            the number of neighbors for each cell, default is 25.
        threads: int
            the number of threads for the cell graph construction, default is 20.
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
                 hidden_dim=256,
                 latent_dim=64,
                 graph_dim=1024,
                 BATCHSIZE=512,
                 max_iter=10,
                 GPU=False,
                 adversarial=True,
                 GCN=True):
        '''
        Set up the training parameters and the model parameters.
        
        parameters
        --------------
        task: str
            the name of this task. It is used to name the output folder.
        dist: str
            the distribution of the single-cell data. Chosen from 'ziln' (zero-inflated log normal), 'zinb' (zero-inflated negative binomial), 'zig' (zero-inflated gamma), and 'nb' (negative binomial).
        device: str
            the device to run the model. If GPU is enabled, the device should be specified. Default is None.
        epoch_iter: int
            the number of epochs for the iterative training process. Default is 10.
        epoch_initial: int
            the number of epochs for the inital iteration. Default is 20.
        lr: float
            the learning rate of the VAE model. Default is 1e-4.
        lr_dis: float
            the learning rate of the discriminator. Default is 5e-4.
        beta: float
            the beta parameter of the beta-VAE. Default is 1.
        hiddem_dim: int
            the hidden dimension of the VAE model. Default is 256.
        latent_dim: int
            the latent dimension of the VAE model. Default is 64.
        graph_dim: int
            the dimension of the GCN layer. Default is 1024.
        BATCHSIZE: int
            the batch size for the model training. Default is 512.
        max_iter: int
            the maximum number of iterations for the model training. Default is 10.
        GPU: bool
            whether to use GPU for the model training. Default is False.
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
        self.hidden_dim = hidden_dim
        self.BATCHSIZE = BATCHSIZE
        self.max_iter = max_iter
        self.GPU = GPU
        if self.GPU:
            assert self.device is not None, "GPU is enabled but device is not specified"
            self.device = torch.device(self.device)
        else:
            self.device = torch.device('cpu')
        #if self.input is not existed then raised error
        if self.input_dim is None:
            raise ValueError('Please use setup_data function to prepare the data first')
        if GCN:
            self.model = VAE(self.input_dim, self.hidden_dim,self.graph_dim, self.latent_dim,beta=self.beta,distribution=self.dist)
        else:
            self.model = Plain_VAE(self.input_dim, self.hidden_dim,self.graph_dim, self.latent_dim,beta=self.beta,distribution=self.dist)
        self.GCN = GCN
        self.adversarial = adversarial
        if self.adversarial:
            self.dis_model = Discriminator(self.input_dim)
        else:
            self.dis_model = None
        self.unagi_trainer = UNAGI_trainer(self.model,self.dis_model,self.task,self.BATCHSIZE,self.epoch_initial,self.epoch_iter,self.device,self.lr, self.lr_dis,GCN=self.GCN,cuda=self.GPU)
    def register_CPO_parameters(self,anchor_neighbors=15, max_neighbors=35, min_neighbors=10, resolution_min=0.8, resolution_max=1.5):
        '''
        The function to register the parameters for the CPO analysis. The parameters will be used to perform the CPO analysis.
        
        parameters
        --------------
        anchor_neighbors: int
            the number of neighbors for each anchor cell.
        max_neighbors: int
            the maximum number of neighbors for each cell.
        min_neighbors: int
            the minimum number of neighbors for each cell.
        resolution_min: float
            the minimum resolution for the Leiden community detection.
        resolution_max: float
            the maximum resolution for the Leiden community detection.
        '''
        self.CPO_parameters = {}
        self.CPO_parameters['anchor_neighbors'] = anchor_neighbors
        self.CPO_parameters['max_neighbors'] = max_neighbors
        self.CPO_parameters['min_neighbors'] = min_neighbors
        self.CPO_parameters['resolution_min'] = resolution_min
        self.CPO_parameters['resolution_max'] = resolution_max
    def register_species(self,species):
        '''
        The function to register the species of the single-cell data.
        
        parameters
        --------------
        species: str
            the species of the single-cell data.
        '''
        if species not in ['human','mouse']:
            raise ValueError('species should be either human or mouse')
        self.species = species

    def register_iDREM_parameters(self,Normalize_data = 'Log_normalize_data', Minimum_Absolute_Log_Ratio_Expression = 0.5, Convergence_Likelihood = 0.001, Minimum_Standard_Deviation = 0.5):
        '''
        The function to register the parameters for the iDREM analysis. The parameters will be used to perform the iDREM analysis.
        
        parameters
        --------------
        Normalize_data: str
            the method to normalize the data. Chosen from 'Log_normalize_data' (log normalize the data), 'Normalize_data' (normalize the data), and 'No_normalize_data' (do not normalize the data).
        Minimum_Absolute_Log_Ratio_Expression: float
            the minimum absolute log ratio expression for the iDREM analysis.
        Convergence_Likelihood: float
            the convergence likelihood for the iDREM analysis.
        Minimum_Standard_Deviation: float
            the minimum standard deviation for the iDREM analysis.
        '''
        
        self.iDREM_parameters = {}
        if Normalize_data not in ['Log_normalize_data','Normalize_data','No_normalize_data']:
            raise ValueError('Normalize_data should be chosen from Log_normalize_data, Normalize_data and No_normalize_data')
        self.iDREM_parameters['Normalize_data'] = Normalize_data
        self.iDREM_parameters['Minimum_Absolute_Log_Ratio_Expression'] = Minimum_Absolute_Log_Ratio_Expression
        self.iDREM_parameters['Convergence_Likelihood'] = Convergence_Likelihood
        self.iDREM_parameters['Minimum_Standard_Deviation'] = Minimum_Standard_Deviation

    def run_UNAGI(self,idrem_dir,CPO=True,resume=False,resume_iteration=None):
        '''
        The function to launch the model training. The model will be trained iteratively. The number of iterations is specified by the `max_iter` parameter in the `setup_training` function.
        
        parameters
        --------------
        idrem_dir: str
            the directory to the iDREM tool which is used to reconstruct the temporal dynamics.
        transcription_factor_file: str
            the directory to the transcription factor file. The transcription factor file is used to perform the CPO analysis.
        '''
        start_iteration = 0
        if resume:
            start_iteration = resume_iteration
        for iteration in range(start_iteration,self.max_iter):
            
            if iteration != 0:
                dir1 = os.path.join(self.data_folder , str(iteration))
                dir2 = os.path.join(self.data_folder , str(iteration)+'/stagedata')
                dir3 = os.path.join(self.data_folder , 'model_save')
                initalcommand = 'mkdir '+ dir1 +' && mkdir '+dir2
                p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
            unagi_runner = UNAGI_runner(self.data_folder,self.ns,iteration,self.unagi_trainer,idrem_dir,adversarial=self.adversarial,GCN = self.GCN)
            unagi_runner.set_up_species(self.species)
            if self.CPO_parameters is not None:
                if type (self.CPO_parameters) != dict:
                    raise ValueError('CPO_parameters should be a dictionary')
                else:
                    unagi_runner.set_up_CPO(anchor_neighbors=self.CPO_parameters['anchor_neighbors'], max_neighbors=self.CPO_parameters['max_neighbors'], min_neighbors=self.CPO_parameters['min_neighbors'], resolution_min=self.CPO_parameters['resolution_min'], resolution_max=self.CPO_parameters['resolution_max'])
            if self.iDREM_parameters is not None:
                if type (self.iDREM_parameters) != dict:
                    raise ValueError('iDREM_parameters should be a dictionary')
                else:
                    unagi_runner.set_up_iDREM(Minimum_Absolute_Log_Ratio_Expression = self.iDREM_parameters['Minimum_Absolute_Log_Ratio_Expression'], Convergence_Likelihood = self.iDREM_parameters['Convergence_Likelihood'], Minimum_Standard_Deviation = self.iDREM_parameters['Minimum_Standard_Deviation'])
            unagi_runner.run(CPO)

    def test_geneweihts(self,iteration,idrem_dir):
        iteration = int(iteration)
        unagi_runner = UNAGI_runner(self.data_folder,self.ns,iteration,self.unagi_trainer,idrem_dir)
        unagi_runner.set_up_species(self.species)
        unagi_runner.load_stage_data()
        unagi_runner.update_gene_weights_table()

    def analyse_UNAGI(self,data_path,iteration,progressionmarker_background_sampling_times,target_dir=None,customized_drug=None,cmap_dir=None):
        '''
        Perform downstream tasks including dynamic markers discoveries, hierarchical markers discoveries, pathway perturbations and compound perturbations.
        
        parameters
        ---------------
        data_path: str
            the directory of the data (h5ad format, e.g. org_dataset.h5ad).
        iteration: int
            the iteration used for analysis.
        progressionmarker_background_sampling_times: int
            the number of times to sample the background cells for dynamic markers discoveries.
        target_dir: str
            the directory to save the results. Default is None.
        customized_drug: str
            the customized drug perturbation list. Default is None.
        cmap_dir: str
            the directory to the cmap database. Default is None.
        '''
        analysts = analyst(data_path,iteration,target_dir=target_dir,customized_drug=customized_drug,cmap_dir=cmap_dir)
        analysts.start_analyse(progressionmarker_background_sampling_times)
        print('The analysis has been done, please check the outputs!')


        

        


        

