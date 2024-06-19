import gc
import os
import shutil
import pickle
import scanpy as sc
import subprocess
import numpy as np
from .utils.analysis_helper import find_overlap_and_assign_direction,calculateDataPathwayOverlapGene,calculateTopPathwayGeneRanking
from .marker_discovery.hierachical_static_markers import get_dataset_hcmarkers
from .perturbations.speedup_perturbation import perturbation
from .marker_discovery.dynamic_markers_helper import get_progressionmarker_background
from .marker_discovery.dynamic_markers import runGetProgressionMarker_one_dist
class analyst:
    '''
    The analyst class is the class to perform downstream analysis. The analyst class will calculate the hierarchical markers, dynamic markers and perform the pathway and drug perturbations. 
    
    parameters
    ----------------
    data_path: str
        the directory of the data (h5ad format, e.g. org_dataset.h5ad).
    iteration: int
        the iteration used for analysis.
    target_dir: str
        the directory to save the results. Default is None.
    customized_drug: str
        the customized drug perturbation list. Default is None.
    cmap_dir: str
        the directory to the cmap database. Default is None.
    '''
    def __init__(self,data_path,iteration,target_dir=None,customized_drug=None,cmap_dir=None):
        self.adata = sc.read(data_path)
        self.data_folder = os.path.dirname(data_path)
        self.adata.uns = pickle.load(open(self.data_folder+'/org_attribute.pkl', 'rb'))
        self.total_stage = len(self.adata.obs['stage'].unique())
        self.customized_drug = customized_drug
        self.cmap_dir = cmap_dir
        self.iteration = iteration
        if target_dir is None:
            self.target_dir = './'+self.data_folder.split('/')[-3]+'_'+str(self.iteration)
            initalcommand = 'mkdir '+ self.target_dir
            p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
        else:
            self.target_dir = target_dir
        self.model_name = self.data_folder.split('/')[-3]+'_'+str(self.iteration)+'.pth'
    def start_analyse(self,progressionmarker_background_sampling):
        '''
        Perform downstream tasks including dynamic markers discoveries, hierarchical markers discoveries, pathway perturbations and compound perturbations.
        
        parameters
        ----------------
        progressionmarker_background_sampling: int
            the number of times to sample the background cells for dynamic markers discoveries.
        '''
        print('calculate hierarchical markers.....')
        hcmarkers= get_dataset_hcmarkers(self.adata,stage_key='stage',cluster_key='leiden',use_rep='umaps')
        print('hierarchical static markers done')
        self.adata = calculateDataPathwayOverlapGene(self.adata)
        print('calculateDataPathwayOverlapGene done')
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        print('calculateTopPathwayGeneRanking done')
        if not os.path.exists(os.path.join(self.target_dir,'idrem')):
            initalcommand = 'cp -r ' + os.path.join(os.path.dirname(self.data_folder),'idremResults') +' '+self.target_dir+'/idrem'
            p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
        initalcommand = 'mkdir '+self.target_dir+'/model_save'+'&& cp ' + os.path.join(os.path.dirname(os.path.dirname(self.data_folder)),'model_save',self.model_name)+' '+self.target_dir+'/model_save/'+self.model_name
        p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
        if self.customized_drug is not None:
            self.adata = find_overlap_and_assign_direction(self.adata, customized_drug=self.customized_drug,cmap_dir=self.cmap_dir)
        else:
            self.adata = find_overlap_and_assign_direction(self.adata,cmap_dir=self.cmap_dir)
        if os.path.exists(os.path.join(self.target_dir,str(progressionmarker_background_sampling)+'progressionmarker_background.npy')):
            progressionmarker_background = np.load(os.path.join(self.target_dir,str(progressionmarker_background_sampling)+'progressionmarker_background.npy'),allow_pickle=True)
            progressionmarker_background = dict(progressionmarker_background.tolist())
        else:
            progressionmarker_background = get_progressionmarker_background(times=progressionmarker_background_sampling,adata= self.adata,total_stage=self.total_stage)
            np.save(os.path.join(self.target_dir,str(progressionmarker_background_sampling)+'progressionmarker_background.npy'),progressionmarker_background)
        self.adata.uns['progressionMarkers'] = runGetProgressionMarker_one_dist(os.path.join(os.path.dirname(self.data_folder),'idremResults'),progressionmarker_background,self.adata.shape[1],cutoff=0.05)
        print('Dynamic markers discovery.....done....')
        gc.collect()
        a = perturbation(self.adata, self.target_dir+'/model_save/'+self.model_name,self.target_dir+'/idrem')
        a.run('pathway',0.5,inplace=True,CUDA=True)
        a.run('drug',0.5,inplace=True)
        a.run('random_background',0.5,inplace=True)
        a.run('online_random_background',0.5,inplace=True)
        a.analysis('pathway',0.5)
        a.analysis('drug',0.5)
        a.adata.uns['hcmarkers'] = hcmarkers #get_dataset_hcmarkers(self.adata,stage_key='stage',cluster_key='leiden',use_rep='umaps')
        with open(os.path.join(self.target_dir,'attribute.pkl'),'wb') as f:
            pickle.dump(a.adata.uns,f)
        del a.adata.uns
        a.adata.obs['leiden'] = a.adata.obs['leiden'].astype(str)
        a.adata.obs['stage'] = a.adata.obs['stage'].astype(str)
        a.adata.obs['ident'] = a.adata.obs['ident'].astype(str)
        a.adata.write(self.target_dir+ '/dataset.h5ad',compression='gzip', compression_opts=9)