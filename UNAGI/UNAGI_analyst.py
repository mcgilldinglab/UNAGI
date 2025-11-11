import gc
import os
import json
import pickle
import scanpy as sc
import shutil
from pathlib import Path
import numpy as np

from VTK.IO.Xdmf3.Testing.Python.VToXLoop import raiseErrorAndExit
from .utils.analysis_helper import find_overlap_and_assign_direction,calculateDataPathwayOverlapGene,calculateTopPathwayGeneRanking,process_customized_drug_database
from .marker_discovery.hierachical_static_markers import get_dataset_hcmarkers
from .perturbations.perturbation import perturbation
from .perturbations.perturbation_centroid import perturbation as perturbation_centroid
from .marker_discovery.dynamic_markers_helper import get_progressionmarker_background
from .marker_discovery.dynamic_markers import runGetProgressionMarker_one_dist
class analyst:
    '''
    The analyst class is the class to perform downstream analysis. The analyst class will calculate the hierarchical markers, dynamic markers and perform the pathway and drug perturbations. 
    
    parameters
    ----------------
    data_path: str
        the directory of the data (h5ad format, e.g. dataset.h5ad).
    iteration: int
        the iteration used for analysis.
    target_dir: str
        the directory to save the results. Default is None.
    customized_drug: str
        the customized drug perturbation list. Default is None.
    cmap_dir: str
        the directory to the cmap database. Default is None.
    '''
    def __init__(self,data_path,iteration,target_dir=None,customized_drug=None,cmap_dir=None,customized_mode = False, training_params = None):
        data_path = Path(data_path)
        self.adata = sc.read(data_path)
        self.data_folder = data_path.parent
        self.adata.uns = pickle.load(open(self.data_folder/'attribute.pkl', 'rb'))
        print(self.adata)
        self.total_stage = len(self.adata.obs['stage'].unique())
        self.customized_drug = customized_drug
        self.cmap_dir = cmap_dir
        self.iteration = iteration
        self.training_params = Path(training_params) if training_params is not None else None
        if target_dir is None:
            self.target_dir = Path(self.data_folder.parts[-3]+'_'+str(self.iteration))
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            self.target_dir = Path(target_dir)
        if training_params is not None:
            training_params_obj = json.load(open(training_params, 'r'))

        else:
            if customized_mode:
                training_params_obj = json.load(open(data_path.parent/'model_save/training_parameters.json','r'))
            else:
                training_params_obj = json.load(open(data_path.parent.parent.parent/'model_save/training_parameters.json','r'))
        self.model_name = training_params_obj['task']+'_'+str(self.iteration)+'.pth'
    def perturbation_analyse_customized_pathway(self,customized_pathway,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu',random_genes=5,random_times=100):
        '''
        Perform perturbation on customized pathway.

        parameters
        ----------------
        customized_pathway: str
            the directory of the customized pathway profile (a npy file).
        perturbed_tracks: str
            the track to perform perturbation. if 'all', all tracks will be used.
        overall_perturbation_analysis: bool
            whether to calculate perturbation scores for all tracks. If False, perturbation scores will be calculated for each track.
        bound: float
            The gene expression changes after perturbation.
        save_csv: str
            the directory to save the perturbation results.
        save_adata: str
            the directory to save the perturbation results.
        CUDA: bool
            whether to use GPU for perturbation.
        device: str
            the device to perform perturbation.
        random_genes: int
            the number of random genes to perform random perturbation.
        random_times: int
            the number of times to build random perturbation score distribution.
        '''
        
        self.adata = calculateDataPathwayOverlapGene(self.adata,customized_pathway=customized_pathway)
        print('calculateDataPathwayOverlapGene done')
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        print('Start perturbation....')
        gc.collect()
        a = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem',config_path=self.training_params)
        a.run('pathway',bound,inplace=True,CUDA=CUDA,device=device)
        a.run('random_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=random_genes,random_times=random_times)
        print('random background done')
        a.analysis('pathway',bound,perturbed_tracks,overall_perturbation_analysis)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['pathway_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_single_gene(self,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu'):
        '''
        Perform perturbation on single gene.

        parameters
        ----------------
        perturbed_tracks: str
            the track to perform perturbation. if 'all', all tracks will be used.
        overall_perturbation_analysis: bool
            whether to calculate perturbation scores for all tracks. If False, perturbation scores will be calculated for each track.
        bound: float
            The gene expression changes after perturbation.
        save_csv: str
            the directory to save the perturbation results.
        save_adata: str
            the directory to save the perturbation results.
        CUDA: bool
            whether to use GPU for perturbation.
        device: str
            the device to perform perturbation.
        random_genes: int
            the number of random genes to perform random perturbation.
        random_times: int
            the number of times to build random perturbation score distribution.
        '''
        
        print('Start perturbation....')
        gc.collect()
        a = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem',config_path=self.training_params)
        a.run('single_gene',bound,inplace=True,CUDA=CUDA,device=device)
        a.adata.uns['random_background_perturbation_deltaD'] = a.adata.uns['single_gene_perturbation_deltaD']
        a.analysis('single_gene',bound,perturbed_tracks,overall_perturbation_analysis)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['single_gene_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_gene_combinatorial(self,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu'):
        '''
        Perform perturbation on gene combinations.

        parameters
        ----------------
        perturbed_tracks: str
            the track to perform perturbation. if 'all', all tracks will be used.
        overall_perturbation_analysis: bool
            whether to calculate perturbation scores for all tracks. If False, perturbation scores will be calculated for each track.
        bound: float
            The gene expression changes after perturbation.
        save_csv: str
            the directory to save the perturbation results.
        save_adata: str
            the directory to save the perturbation results.
        CUDA: bool
            whether to use GPU for perturbation.
        device: str
            the device to perform perturbation.
        random_genes: int
            the number of random genes to perform random perturbation.
        random_times: int
            the number of times to build random perturbation score distribution.
        '''
        
        print('Start perturbation....')
        gc.collect()
        a = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem',config_path=self.training_params)
        a.run('two_genes',bound,inplace=True,CUDA=CUDA,device=device)
        a.run('random_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=2,random_times=20)
        print('random background done')
        a.analysis('two_genes',bound,perturbed_tracks,overall_perturbation_analysis)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['two_genes_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_customized_drug(self,customized_drug,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=True,device='cuda:0',random_genes=2,random_times=100):
        '''
        Perform perturbation on customized drug.

        parameters
        ----------------
        customized_drug: str
            the directory of the customized drug profile (a npy file).
        perturbed_tracks: str
            the track to perform perturbation. if 'all', all tracks will be used.
        overall_perturbation_analysis: bool
            whether to calculate perturbation scores for all tracks. If False, perturbation scores will be calculated for each track.
        bound: float
            The gene expression changes after perturbation.
        save_csv: str
            the directory to save the perturbation results.
        save_adata: str
            the directory to save the perturbation results.
        CUDA: bool
            whether to use GPU for perturbation.
        device: str
            the device to perform perturbation.
        random_genes: int
            the number of random genes to perform random perturbation.
        random_times: int
            the number of times to build random perturbation score distribution.
        
        '''

        self.adata = process_customized_drug_database(self.adata, customized_drug=customized_drug)
        print('Start perturbation....')
        gc.collect()
        a = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem',config_path=self.training_params)
        a.run('drug',bound,inplace=True,CUDA=CUDA,device=device)
        print('drug perturabtion done')
        a.run('random_drug_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=random_genes,random_times=random_times)
        print('random background done')
        a.analysis('drug',bound,perturbed_tracks,overall_perturbation_analysis=overall_perturbation_analysis)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['drug_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def get_median_random_gene(self, data_drug_overlapped):
        '''
        Get the median number of random genes for each drug. The median number of random genes is used to perform random perturbation.

        parameters
        ----------------
        data_drug_overlapped: dict
            the dictionary of drug and the overlapped genes.
        '''
        each_drug_data_overlappings = []
        for each in data_drug_overlapped.keys():

            each_drug_data_overlappings.append(len(data_drug_overlapped[each]))
        return np.median(each_drug_data_overlappings)
    def cmap_overlapped_genes(self,data_drug_overlapped):
        output = []
        for each in data_drug_overlapped.keys():
            for each_gene in data_drug_overlapped[each]:
                output.append(each_gene.split(':')[0])
            break
        return output
        
    def start_analyse(self,progressionmarker_background_sampling,run_perturbation,random_times, ignore_dynamic_markers=False, ignore_hcmarkers=False,customized_pathway=None,perturb_change=0.5,overall_perturbation_analysis=True,perturbed_tracks='all',ignore_pathway_perturabtion=False,ignore_drug_perturabtion=False,centroid=False):
        '''
        Perform downstream tasks including dynamic markers discoveries, hierarchical markers discoveries, pathway perturbations and compound perturbations.
        
        parameters
        ----------------
        progressionmarker_background_sampling: int
            the number of times to sample the background cells for dynamic markers discoveries.
        run_perturbation: bool
            whether to perform perturbation analysis.
        perturb_change: float
            The gene expression changes after perturbation..
        overall_perturbation_analysis: bool
            whether to use all tracks for perturbation analysis.
        perturbed_tracks: str
            the track to perform perturbation.
        centroid: bool
            whether to use centroid perturbation. If True, the perturbation will be performed on the centroid of each cluster. Using centroid perturbation will speed up the perturbation process, but may lose some information.
        ignore_dynamic_markers: bool
            whether to ignore dynamic markers discoveries.
        ignore_hcmarkers: bool
            whether to ignore hierarchical markers discoveries.
        customized_pathway: str
            the directory of the customized pathway profile (a npy file). If None, the default pathway profile (only REACTOME database) will be used.
        ignore_pathway_perturabtion: bool
            whether to ignore pathway perturbation.
        ignore_drug_perturabtion: bool
            whether to ignore drug perturbation.
        '''
        
        if not ignore_hcmarkers:
            print('calculate hierarchical markers.....')
            hcmarkers= get_dataset_hcmarkers(self.adata,stage_key='stage',cluster_key='leiden',use_rep='umaps')
            print('hierarchical static markers done')

        if customized_pathway is not None:
            self.adata = calculateDataPathwayOverlapGene(self.adata,customized_pathway=customized_pathway)
        else:
            self.adata = calculateDataPathwayOverlapGene(self.adata)
        print('calculateDataPathwayOverlapGene done')
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        print('calculateTopPathwayGeneRanking done')
        if not os.path.exists(self.target_dir/'idrem'):
            shutil.copytree(self.data_folder.parent/'idremResults', self.target_dir/'idrem')
        
        (self.target_dir/'model_save').mkdir(parents=True, exist_ok=True)
        shutil.copy(self.data_folder.parent.parent/'model_save'/self.model_name, self.target_dir/'model_save'/self.model_name)
        shutil.copy(self.data_folder.parent.parent/'model_save'/'training_parameters.json', self.target_dir/'model_save'/'training_parameters.json')
        #stop 10 seconds to ensure the model is copied
        import time
        time.sleep(10)
        if not ignore_dynamic_markers:

            if os.path.exists(self.target_dir/(str(progressionmarker_background_sampling)+'progressionmarker_background.npy')):
                progressionmarker_background = np.load(self.target_dir/(str(progressionmarker_background_sampling)+'progressionmarker_background.npy'),allow_pickle=True)
                progressionmarker_background = dict(progressionmarker_background.tolist())
            else:
                progressionmarker_background = get_progressionmarker_background(times=progressionmarker_background_sampling,adata= self.adata,total_stage=self.total_stage)
                np.save(self.target_dir/(str(progressionmarker_background_sampling)+'progressionmarker_background.npy'),progressionmarker_background)
            self.adata.uns['progressionMarkers'] = runGetProgressionMarker_one_dist(os.path.join(os.path.dirname(self.data_folder),'idremResults'),progressionmarker_background,self.adata.shape[1],cutoff=0.05)
            print('Dynamic markers discovery.....done....')
        if not centroid:
            perturbation_runner = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        else:
            perturbation_runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        if run_perturbation:
            direction_flag = False
            if not ignore_drug_perturabtion:
                try:
                    temp_drug = np.load(self.customized_drug,allow_pickle=True).item()
                    if ':' in temp_drug[list(temp_drug.keys())[0]][0]:
                        direction_flag = True
                except:
                    raise ValueError('Invalid customized drug format.')
                if direction_flag:
                    customized_direction = self.customized_drug
                if self.customized_drug is not None:
                    self.adata = find_overlap_and_assign_direction(self.adata, customized_drug=self.customized_drug,customized_direction=customized_direction)
                else:
                    if self.cmap_dir is not None:

                        self.adata = find_overlap_and_assign_direction(self.adata,cmap_dir=self.cmap_dir)
                    else:
                        raise ValueError('Please provide a cmap_dir or a customized drug database.')
            print('Start perturbation....')
            gc.collect()
            if not ignore_pathway_perturabtion:
                perturbation_runner.run('random_pathway_background',perturb_change,inplace=True,CUDA=True)
                perturbation_runner.run('pathway',perturb_change,inplace=True,CUDA=True)
                if perturbed_tracks != 'all':
                    perturbation_runner.analysis('pathway',perturb_change,perturbed_tracks,overall_perturbation_analysis=False)
                else:
                    perturbation_runner.analysis('pathway',perturb_change,perturbed_tracks,overall_perturbation_analysis)
            else:
                print('Ignore pathway perturbation!')
            if not ignore_drug_perturabtion:
                perturbation_runner.run('drug',perturb_change,inplace=True)
                print('drug perturabtion done')
                
                if self.customized_drug is not None:
                    perturbation_runner.run('random_drug_background',perturb_change,inplace=True,random_times = random_times, random_genes=self.get_median_random_gene(perturbation_runner.adata.uns['data_drug_overlap_genes']))
                else:
                    random_gene = self.cmap_overlapped_genes(perturbation_runner.adata.uns['data_drug_overlap_genes'])
                    random_genes = [] 
                    import random
                    
                    for time in range(progressionmarker_background_sampling):
                        random.seed(time)
                        choices = [random.choice(['+','-']) for _ in range(len(random_gene))]
                        random_genes.append([random_gene[i] + ':' + choices[i] for i in range(len(random_gene))])
                    perturbation_runner.adata.uns['cmap_random_genes'] = random_genes
                    perturbation_runner.run('random_drug_background',perturb_change,inplace=True,random_genes=random_genes)
                if perturbed_tracks!='all':
                    perturbation_runner.analysis('drug',perturb_change,perturbed_tracks,overall_perturbation_analysis=False)
                else:
                    perturbation_runner.analysis('drug',perturb_change,perturbed_tracks,overall_perturbation_analysis)
                print('analysis of drug perturbation')
            else:
                print('Ignore drug perturbation!')
        if not ignore_hcmarkers:
            perturbation_runner.adata.uns['hcmarkers'] = hcmarkers 
        with open(os.path.join(self.target_dir,'attribute.pkl'),'wb') as f:
            pickle.dump(perturbation_runner.adata.uns,f)
        del perturbation_runner.adata.uns
        perturbation_runner.adata.obs['leiden'] = perturbation_runner.adata.obs['leiden'].astype(str)
        perturbation_runner.adata.obs['stage'] = perturbation_runner.adata.obs['stage'].astype(str)
        perturbation_runner.adata.obs['ident'] = perturbation_runner.adata.obs['ident'].astype(str)
        perturbation_runner.adata.write(self.target_dir/'dataset.h5ad',compression='gzip', compression_opts=9)
        # return a.adata
    def drug_perturbation_analysis(self,tracks,perturb_change, centroid=False):
        key = perturb_change
        track_results = {}
        if not centroid:
            perturbation_runner = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        else:
            perturbation_runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        if tracks == 'individual':
            tracks_head = []
            for each in list(self.adata.uns['drug_perturbation_deltaD'][str(key)].keys()):
                each = each.split('-')[0]
                if each not in tracks_head:
                    tracks_head.append(each)
            if  'overall' in self.adata.uns['drug_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            for each_track in tracks_head:
                print('Processing track %s'%(each_track))
                perturbation_runner.analysis('drug',key,each_track,overall_perturbation_analysis=True)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()

            del perturbation_runner.adata.uns['drug_perturbation_score']
            perturbation_runner.adata.uns['drug_perturbation_score'] = {}
            perturbation_runner.adata.uns['drug_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['drug_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['drug_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['drug_perturbation_score'][str(1/key)][each] = track_results[each]
        else:
            track_results = {}
            if  'overall' in self.adata.uns['drug_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('drug',key, tracks,overall_perturbation_analysis=True)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            del perturbation_runner.adata.uns['drug_perturbation_score']
            perturbation_runner.adata.uns['drug_perturbation_score'] = {}
            perturbation_runner.adata.uns['drug_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['drug_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['drug_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['drug_perturbation_score'][str(1/key)][each] = track_results[each]
        return perturbation_runner.adata
    def pathway_perturbation_analysis(self,tracks,perturb_change, centroid=False):
        key = perturb_change
        track_results = {}
        if not centroid:
            perturbation_runner = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        else:
            perturbation_runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        if tracks == 'individual':
            tracks_head = []
            for each in list(self.adata.uns['pathway_perturbation_deltaD'][str(key)].keys()):
                each = each.split('-')[0]
                if each not in tracks_head:
                    tracks_head.append(each)
            if  'overall' in self.adata.uns['pathway_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            for each_track in tracks_head:
                print('Processing track %s'%(each_track))
                perturbation_runner.analysis('pathway',key,each_track,overall_perturbation_analysis=True)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()

            del perturbation_runner.adata.uns['pathway_perturbation_score']
            perturbation_runner.adata.uns['pathway_perturbation_score'] = {}
            perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['pathway_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['pathway_perturbation_score'][str(1/key)][each] = track_results[each]
        else:
            track_results = {}
            if  'overall' in self.adata.uns['pathway_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('pathway',key, tracks,overall_perturbation_analysis=True)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            del perturbation_runner.adata.uns['pathway_perturbation_score']
            perturbation_runner.adata.uns['pathway_perturbation_score'] = {}
            perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['pathway_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['pathway_perturbation_score'][str(1/key)][each] = track_results[each]
        return perturbation_runner.adata
    def single_gene_perturbation_analysis(self,tracks,perturb_change, centroid=False):
        key = perturb_change
        track_results = {}
        if not centroid:
            perturbation_runner = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        else:
            perturbation_runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        if tracks == 'individual':
            tracks_head = []
            for each in list(self.adata.uns['single_gene_perturbation_deltaD'][str(key)].keys()):
                each = each.split('-')[0]
                if each not in tracks_head:
                    tracks_head.append(each)
            if  'overall' in self.adata.uns['single_gene_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            for each_track in tracks_head:
                print('Processing track %s'%(each_track))
                perturbation_runner.analysis('single_gene',key,each_track,overall_perturbation_analysis=True)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()

            del perturbation_runner.adata.uns['single_gene_perturbation_score']
            perturbation_runner.adata.uns['single_gene_perturbation_score'] = {}
            perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['single_gene_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['single_gene_perturbation_score'][str(1/key)][each] = track_results[each]
        else:
            track_results = {}
            if  'overall' in self.adata.uns['single_gene_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('single_gene',key, tracks,overall_perturbation_analysis=True)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            del perturbation_runner.adata.uns['single_gene_perturbation_score']
            perturbation_runner.adata.uns['single_gene_perturbation_score'] = {}
            perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['single_gene_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['single_gene_perturbation_score'][str(1/key)][each] = track_results[each]
        return perturbation_runner.adata
    def gene_combinatorial_perturbation_analysis(self,tracks,perturb_change, centroid=False):
        key = perturb_change
        track_results = {}
        if not centroid:
            perturbation_runner = perturbation(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        else:
            perturbation_runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        if tracks == 'individual':
            tracks_head = []
            for each in list(self.adata.uns['two_genes_perturbation_deltaD'][str(key)].keys()):
                each = each.split('-')[0]
                if each not in tracks_head:
                    tracks_head.append(each)
            if  'overall' in self.adata.uns['two_genes_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            for each_track in tracks_head:
                print('Processing track %s'%(each_track))
                perturbation_runner.analysis('two_genes',key,each_track,overall_perturbation_analysis=True)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()

            del perturbation_runner.adata.uns['two_genes_perturbation_score']
            perturbation_runner.adata.uns['two_genes_perturbation_score'] = {}
            perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['two_genes_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['two_genes_perturbation_score'][str(1/key)][each] = track_results[each]
        else:
            track_results = {}
            if  'overall' in self.adata.uns['two_genes_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('two_genes',key, tracks,overall_perturbation_analysis=True)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            del perturbation_runner.adata.uns['two_genes_perturbation_score']
            perturbation_runner.adata.uns['two_genes_perturbation_score'] = {}
            perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)] = {}
            perturbation_runner.adata.uns['two_genes_perturbation_score'][str(1/key)] = {}
            for each in track_results.keys():  
                perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)][each] = track_results[each]
                perturbation_runner.adata.uns['two_genes_perturbation_score'][str(1/key)][each] = track_results[each]
        return perturbation_runner.adata