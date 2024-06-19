import subprocess
from ..utils.gcn_utils import  setup_graph
from ..utils.CPO_utils import get_neighbors, auto_resolution
from ..utils.attribute_utils import saveRep,get_all_adj_adata,mergeAdata,updateAttributes,get_data_file_path
from ..dynamic_graphs.buildGraph import getandUpadateEdges
import gc
import scanpy as sc
import numpy as np
import os
from ..dynamic_regulatory_networks.processIDREM import getClusterPaths, getClusterIdrem, runIdrem
from ..dynamic_regulatory_networks.processTFs import getTFs, getTargetGenes, matchTFandTGWithFoldChange, updataGeneTablesWithDecay
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
class UNAGI_runner:
    '''
    The UNAGI_runner class is used to set up the hyperparameters to run iDREM, find clustering optimal parameters and run the UNAGI model . It takes the following 
    
    parameters
    ------------
    data_path: the path to the data
    total_stage: the total number of time-series stages
    iteration: the total iteration to run the UNAGI model
    trainer: the trainer class to train the UNAGI model
    idrem_dir: the directory of the idrem software
    '''
    def __init__(self,data_path,total_stage,iteration,trainer,idrem_dir,adversarial=True,GCN=True,connect_edges_cutoff=0.05):
        self.data_path = data_path
        self.total_stage = total_stage
        self.iteration = iteration
        self.trainer = trainer
        self.resolutions = None
        self.idrem_dir = idrem_dir
        self.neighbor_parameters = None
        self.setup_CPO = False
        self.species = None
        self.setup_IDREM = False
        self.adversarial = adversarial
        self.GCN = GCN
        self.connect_edges_cutoff = connect_edges_cutoff
    def load_stage_data(self):
        '''
        Load the stage data from the data_path. The stage data will be stored in the adata_stages list. The all_in_one adata will be used for the UNAGI model training.
        '''
        stageadata = []
        for i in range(self.total_stage):
            if self.iteration != 0:
                adata = sc.read(os.path.join(self.data_path, '%d/stagedata/%d.h5ad'%(self.iteration-1,i)))
            elif self.iteration == 0:
                adata = sc.read(os.path.join(self.data_path, '%d.h5ad'%(i)))
            if 'leiden_colors' in adata.uns:
                del adata.uns['leiden_colors']
            stageadata.append(adata)
        self.all_in_one = get_all_adj_adata(stageadata)
        self.adata_stages = stageadata
        self.genenames = np.array(list(self.adata_stages[0].var.index.values))
    def annotate_stage_data(self, adata,stage,CPO):
        '''
        Retreive the latent representations of given single cell data. Performing clusterings, generating the UMAPs, annotating the cell types and adding the top genes and cell types attributes.

        Parameters
        ------------
        adata: anndata
            the single cell data.
        stage: int
            the stage of the single cell data.

        return
        ------------
        adata: anndata
            the annotated single cell data.
        averageValue: list
            the average value of each cluster.
        reps: list
            the latent representations of the single cell data.
        '''
        z_locs, z_scales, cell_embeddings = self.trainer.get_latent_representation(adata,self.iteration,self.data_path)
        adata.obsm['z'] = cell_embeddings
        if self.neighbor_parameters is None:
            sc.pp.neighbors(adata, use_rep="z",n_neighbors=50,method='umap')
            return adata
        if 'connectivities' in adata.obsp.keys():
            del adata.obsp['connectivities']
            del adata.obsp['distances']
        z_adj = kneighbors_graph(adata.obsm['z'], self.neighbor_parameters[stage], mode='connectivity', include_self=True,n_jobs=20)
        adata.obsp['connectivities'] = z_adj
        adata.obsp['distances'] = kneighbors_graph(adata.obsm['z'], self.neighbor_parameters[stage], mode='distance', include_self=True,n_jobs=20)
        if CPO is False:
            sc.pp.neighbors(adata, use_rep="z",n_neighbors=self.neighbor_parameters[stage],method='umap')
        sc.tl.leiden(adata,resolution = self.resolutions[stage])
        sc.tl.paga(adata)
        sc.pl.paga(adata,show=False)
        # sc.tl.umap(adata, min_dist=0.25/self.resolution_coefficient*len(adata),init_pos='paga')
        sc.tl.umap(adata, min_dist=0.05,init_pos='paga')
        rep=[z_locs, z_scales]
        adata.obs['ident'] = 'None'
        adata.obs['leiden'] = adata.obs['leiden'].astype(str)
        sc.pl.umap(adata,color='leiden',show=False)
        adata,averageValue,reps = updateAttributes(adata,rep)#add top genes, cell types attributes and get average cluster value
  
        adata.obs['leiden']=adata.obs['leiden'].astype(str)
        if self.iteration == 0:
            allzeros = np.zeros_like(adata.X)
            allzeros = csr_matrix(allzeros)
            adata.layers['geneWeight'] = allzeros
        sc.pl.umap(adata,color='ident',show=False)
        adata.write(os.path.join(self.data_path,str(self.iteration)+'/stagedata/%d.h5ad'%stage),compression='gzip',compression_opts=9)
        return adata,averageValue,reps
    
    def set_up_CPO(self, anchor_neighbors, max_neighbors, min_neighbors, resolution_min, resolution_max):
        '''
        Set up the parameters for finding the clustering optimal parameters.

        Parameters
        ------------
        anchor_neighbors: int
            the number of neighbors for the anchor cells.
        max_neighbors: int
            the maximum number of neighbors for the single cell data.
        min_neighbors: int
            the minimum number of neighbors for the single cell data.
        resolution_min: float
            the minimum resolution for the single cell data.
        resolution_max: float   
            the maximum resolution for the single cell data.
        '''
        self.setup_CPO = True
        self.anchor_neighbors = anchor_neighbors
        self.max_neighbors = max_neighbors
        self.min_neighbors = min_neighbors
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max

    def run_CPO(self):
        '''
        Find the clustering optimal parameters for the single cell data.
        '''
        max_adata_cells = 0
        num_cells = []
        for each in self.adata_stages:
            num_cells.append(each.shape[0])
            if len(each) > max_adata_cells:
                max_adata_cells = len(each)
        self.resolution_coefficient = max_adata_cells
        for i in range(0,len(self.adata_stages)):
            self.adata_stages[i] = self.annotate_stage_data(self.adata_stages[i], i,CPO=True)
        if not self.setup_CPO:
            print('CPO parameters are not set up, using default parameters')
            print('anchor_neighbors: 15, max_neighbors: 35, min_neighbors: 10, resolution_min: 0.8, resolution_max: 1.5')
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells,anchor_neighbors=15,max_neighbors=35,min_neighbors=10)
            self.resolutions,_ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters, 0.8, 1.5)
        else:
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells,anchor_neighbors=self.anchor_neighbors,max_neighbors=self.max_neighbors,min_neighbors=self.min_neighbors)
            self.resolutions,_ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters, self.resolution_min, self.resolution_max)
    
    def update_cell_attributes(self,CPO):
        '''
        Update and save the cell attributes including the top genes, cell types and latent representations.
        '''
        self.averageValues = []
        reps = []
        for i in range(0,len(self.adata_stages)):
            adata = self.adata_stages[i]
            # print(adata.X.shape)
            adata.uns['topGene']={}
            adata.uns['clusterType']={}
            adata.uns['rep']={}
            adata,averageValue,rep = self.annotate_stage_data(adata,i,CPO)
            gc.collect()
            reps.append(rep)
            averageValue = np.array(averageValue)
            self.averageValues.append(averageValue)
        self.averageValues = np.array(self.averageValues,dtype=object)
        np.save(os.path.join(self.data_path, '%d/averageValues.npy'%self.iteration),self.averageValues)
        saveRep(reps,self.data_path,self.iteration)
    def build_temporal_dynamics_graph(self):
        '''
        Build the temporal dynamics graph.
        '''
        self.edges = getandUpadateEdges(self.total_stage,self.data_path,self.iteration,self.connect_edges_cutoff)
    def set_up_IDREM(self,Minimum_Absolute_Log_Ratio_Expression, Convergence_Likelihood, Minimum_Standard_Deviation):
        '''
        Set up the parameters for running the iDREM software.

        Parameters
        ------------
        Minimum_Absolute_Log_Ratio_Expression: float
            the minimum absolute log ratio expression.
        Convergence_Likelihood: float
            the convergence likelihood.
        Minimum_Standard_Deviation: float
            the minimum standard deviation.
        '''
        self.setup_IDREM = True
        self.Minimum_Absolute_Log_Ratio_Expression = Minimum_Absolute_Log_Ratio_Expression
        self.Convergence_Likelihood = Convergence_Likelihood
        self.Minimum_Standard_Deviation = Minimum_Standard_Deviation
    def set_up_species(self, species):
        '''
        Set up the species for running the iDREM software.

        Parameters
        ------------
        species: str
            the species of the single cell data.
        '''

        print('Species: Running on %s data'%species)
        self.species = species
    def run_IDREM(self):
        '''
        Run the iDREM software.
        '''
        averageValues = np.load(os.path.join(self.data_path, '%d/averageValues.npy'%self.iteration),allow_pickle=True)
        paths = getClusterPaths(self.edges,self.total_stage)
        idrem= getClusterIdrem(paths,averageValues,self.total_stage)
        paths = [each for each in paths.values() if len(each) == self.total_stage]
        # print(paths)
        idrem = np.array(idrem)
        self.genenames =np.array(list(self.adata_stages[0].var.index.values))
        if not self.setup_IDREM:
            print('IDREM parameters are not set up, using default parameters')
            print('Minimum_Absolute_Log_Ratio_Expression: 0.5, Convergence_Likelihood: 0.001, Minimum_Standard_Deviation: 0.5')
            if self.species is None:
                print('Human species is used as default')
                runIdrem(paths,self.data_path,idrem,self.genenames,self.iteration,self.idrem_dir)
            else:
                runIdrem(paths,self.data_path,idrem,self.genenames,self.iteration,self.idrem_dir,species=self.species)
        else:
            if self.species is None:
                print('Human species is used as default')
                runIdrem(paths,self.data_path,idrem,self.genenames,self.iteration,self.idrem_dir,Minimum_Absolute_Log_Ratio_Expression=self.Minimum_Absolute_Log_Ratio_Expression, Convergence_Likelihood=self.Convergence_Likelihood, Minimum_Standard_Deviation=self.Minimum_Standard_Deviation)
            else:
                runIdrem(paths,self.data_path,idrem,self.genenames,self.iteration,self.idrem_dir,species=self.species,Minimum_Absolute_Log_Ratio_Expression=self.Minimum_Absolute_Log_Ratio_Expression, Convergence_Likelihood=self.Convergence_Likelihood, Minimum_Standard_Deviation=self.Minimum_Standard_Deviation)
            
    def update_gene_weights_table(self,topN=100):
        '''
        Update the gene weights table.

        Parameters
        ------------
        topN: int
            the number of top genes to be selected.
        
        '''
        TFs = getTFs(os.path.join(self.data_path,str(self.iteration)+'/'+'idremResults'+'/'),total_stage=self.total_stage)
        np.save('test_TFs.npy',np.array(TFs,dtype=object))
        scope = getTargetGenes(os.path.join(self.data_path,str(self.iteration)+'/'+'idremResults'+'/'),topN)
        np.save('test_scope.npy',np.array(scope,dtype=object))
        self.averageValues = np.load(os.path.join(self.data_path, '%d/averageValues.npy'%self.iteration),allow_pickle=True)
        p = matchTFandTGWithFoldChange(TFs,scope,self.averageValues,get_data_file_path('human_encode.txt'),self.genenames,self.total_stage)
        np.save('test_p.npy',np.array(p,dtype=object))
        #np.save('../data/mes/'+str(iteration)+'/tfinfo.npy',np.array(p))
        updateLoss = updataGeneTablesWithDecay(self.data_path,str(self.iteration),p,self.total_stage)
    def build_iteration_dataset(self):
        '''
        Build the iteration dataset.
        '''
        mergeAdata(os.path.join(self.data_path,str(self.iteration)),total_stages=self.total_stage)
       
    def run(self,CPO):
        '''
        Run the UNAGI pipeline.
        '''
        self.load_stage_data()
        if self.iteration == 0:
            is_iterative = False
        else:
            is_iterative = True
        self.trainer.train(self.all_in_one,self.iteration,target_dir=self.data_path,adversarial=self.adversarial,is_iterative=is_iterative)
        if CPO:
            self.run_CPO()
            
        else:
            self.resolutions = [1.0]*self.total_stage
            self.neighbor_parameters = [30]*self.total_stage
        self.update_cell_attributes(CPO)
        self.build_temporal_dynamics_graph()
        self.run_IDREM()
        self.update_gene_weights_table()
        self.build_iteration_dataset()


