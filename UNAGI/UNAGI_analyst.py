import copy
import gc
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from .utils.analysis_helper import find_overlap_and_assign_direction,calculateDataPathwayOverlapGene,calculateTopPathwayGeneRanking,process_customized_drug_database
from .utils.progress_logger import ProgressLogger
from .marker_discovery.hierachical_static_markers import get_dataset_hcmarkers
from .perturbations.perturbation import perturbation
from .perturbations.perturbation_centroid import perturbation as perturbation_centroid
from .perturbations.exact_parallel_executor import default_cpu_workers, run_exact_parallel_delta
from .perturbations.direction_utils import infer_direction_mode, store_directional_result
from .marker_discovery.dynamic_markers_helper import get_progressionmarker_background
from .marker_discovery.dynamic_markers import runGetProgressionMarker_one_dist


PERTURBATION_STAGE_CACHE_VERSION = 2
SUPPORTED_PERTURBATION_SCHEDULES = {'single_gpu_optimized', 'resource_aware'}
SUPPORTED_DYNAMIC_MARKER_STRATEGIES = {'legacy', 'optimized'}


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
        data_path = Path(data_path).expanduser().resolve()
        self.adata = sc.read(data_path)
        self.data_path = data_path
        self.data_folder = data_path.parent
        self.attribute_path = self.data_folder/'attribute.pkl'
        self.adata.uns = pickle.load(open(self.attribute_path, 'rb'))
        print(self.adata)
        self.total_stage = len(self.adata.obs['stage'].unique())
        self.customized_drug = customized_drug
        self.cmap_dir = cmap_dir
        self.iteration = iteration
        self.training_params = Path(training_params).expanduser().resolve() if training_params is not None else None
        if target_dir is None:
            self.target_dir = Path(self.data_folder.parts[-3]+'_'+str(self.iteration))
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            self.target_dir = Path(target_dir).expanduser().resolve()
        if training_params is not None:
            training_params_obj = json.load(open(training_params, 'r'))
            self.training_params_path = self.training_params

        else:
            if customized_mode:
                self.training_params_path = data_path.parent/'model_save'/'training_parameters.json'
                training_params_obj = json.load(open(self.training_params_path,'r'))
            else:
                self.training_params_path = data_path.parent.parent.parent/'model_save'/'training_parameters.json'
                training_params_obj = json.load(open(self.training_params_path,'r'))
        self.model_name = training_params_obj['task']+'_'+str(self.iteration)+'.pth'
        self.model_source_path = self.training_params_path.parent / self.model_name
    def perturbation_analyse_customized_pathway(self,customized_pathway,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu',random_genes=5,random_times=100,direction_mode='two_sided',perturb_stage=None):
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
        a.run('pathway',bound,inplace=True,CUDA=CUDA,device=device,direction_mode=direction_mode)
        a.run('random_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=random_genes,random_times=random_times,direction_mode=direction_mode)
        print('random background done')
        stage_info = self._resolve_perturb_stage(perturb_stage)
        a.analysis(
            'pathway',
            bound,
            perturbed_tracks,
            overall_perturbation_analysis,
            stage=stage_info['index'] if stage_info is not None else None,
            direction_mode=direction_mode,
        )
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['pathway_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_single_gene(self,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu',direction_mode='two_sided'):
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
        a.run('single_gene',bound,inplace=True,CUDA=CUDA,device=device,direction_mode=direction_mode)
        resolved_mode = self._copy_single_gene_background(a.adata, bound, direction_mode)
        a.analysis('single_gene',bound,perturbed_tracks,overall_perturbation_analysis,direction_mode=resolved_mode)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['single_gene_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_gene_combinatorial(self,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=False,device='cpu',direction_mode='two_sided'):
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
        a.run('two_genes',bound,inplace=True,CUDA=CUDA,device=device,direction_mode=direction_mode)
        a.run('random_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=2,random_times=20,direction_mode=direction_mode)
        print('random background done')
        a.analysis('two_genes',bound,perturbed_tracks,overall_perturbation_analysis,direction_mode=direction_mode)
        print('Finish results analysis')
        if save_csv is not None:
            a.uns['two_genes_perturbation'].to_csv(Path(save_csv))
        if save_adata is not None:
            a.adata.write(Path(save_adata),compression='gzip', compression_opts=9)

    def perturbation_analyse_customized_drug(self,customized_drug,perturbed_tracks='all',overall_perturbation_analysis=True,bound=0.5,save_csv = None,save_adata = None,CUDA=True,device='cuda:0',random_genes=2,random_times=100,direction_mode='two_sided',perturb_stage=None):
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
        a.run('drug',bound,inplace=True,CUDA=CUDA,device=device,direction_mode=direction_mode)
        print('drug perturabtion done')
        a.run('random_drug_background',bound,inplace=True,CUDA=CUDA,device=device,random_genes=random_genes,random_times=random_times,direction_mode=direction_mode)
        print('random background done')
        stage_info = self._resolve_perturb_stage(perturb_stage)
        a.analysis(
            'drug',
            bound,
            perturbed_tracks,
            overall_perturbation_analysis=overall_perturbation_analysis,
            stage=stage_info['index'] if stage_info is not None else None,
            direction_mode=direction_mode,
        )
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

    def _normalize_perturbation_schedule(self, perturbation_schedule):
        schedule = str(perturbation_schedule or 'single_gpu_optimized').strip().lower()
        if schedule not in SUPPORTED_PERTURBATION_SCHEDULES:
            raise ValueError(
                "perturbation_schedule must be one of: single_gpu_optimized, resource_aware"
            )
        if schedule == 'resource_aware':
            raise NotImplementedError(
                "perturbation_schedule=resource_aware is reserved for a future multi-process, multi-GPU implementation."
            )
        return schedule
    @staticmethod
    def _gpu_available():
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    def _resolve_perturb_execution_backend(self, *, centroid, cpu_workers=None):
        # Prefer GPU whenever it is available. cpu_workers controls CPU
        # parallelism for CPU execution and GPU fallback, but it does not
        # disable GPU by itself.
        gpu_available = self._gpu_available()
        if centroid:
            backend = 'gpu_centroid' if gpu_available else 'cpu_centroid'
        else:
            backend = 'gpu_exact' if gpu_available else 'cpu_exact_parallel'
        return {
            'backend': backend,
            'use_cuda': backend.startswith('gpu_'),
            'device': 'cuda:0' if backend.startswith('gpu_') else 'cpu',
            'cpu_workers': default_cpu_workers(cpu_workers),
        }

    @staticmethod
    def _is_gpu_execution_backend(execution_backend):
        return str(execution_backend).startswith('gpu_')

    @staticmethod
    def _cpu_fallback_backend(execution_backend):
        return 'cpu_centroid' if str(execution_backend) == 'gpu_centroid' else 'cpu_exact_parallel'

    @staticmethod
    def _is_gpu_runtime_error(exc):
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, torch.OutOfMemoryError):
                return True
            message = str(current).lower()
            if any(
                token in message
                for token in (
                    'cuda',
                    'cublas',
                    'cudnn',
                    'device-side assert',
                    'no kernel image',
                    'not compiled with cuda',
                    'invalid device',
                    'not compatible with the current pytorch installation',
                    'sm_',
                )
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _normalize_dynamic_marker_strategy(self, dynamic_marker_strategy):
        strategy = str(dynamic_marker_strategy or 'optimized').strip().lower()
        if strategy not in SUPPORTED_DYNAMIC_MARKER_STRATEGIES:
            raise ValueError(
                'dynamic_marker_strategy must be one of: legacy, optimized'
            )
        return strategy

    def _path_fingerprint(self, path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return {'path': str(path), 'mtime_ns': None, 'size': None}
        stat = path.stat()
        return {'path': str(path), 'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size}

    def _cache_root(self):
        root = self.target_dir / 'perturbation_stage_cache'
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _cache_source_identity(self):
        return {
            'dataset': self._path_fingerprint(self.data_path),
            'attribute': self._path_fingerprint(self.attribute_path),
            'training_params': self._path_fingerprint(self.training_params_path),
            'model': self._path_fingerprint(self.model_source_path),
            'iteration': int(self.iteration),
        }

    def _normalize_track_request(self, perturbed_tracks):
        if perturbed_tracks == 'all':
            return 'all'
        if isinstance(perturbed_tracks, (list, tuple, set)):
            return [str(each) for each in perturbed_tracks]
        return str(perturbed_tracks)

    def _stage_order(self):
        ordered = []
        seen = set()
        for value in self.adata.obs['stage'].astype(str).tolist():
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _resolve_perturb_stage(self, perturb_stage):
        if perturb_stage in (None, '', 'all'):
            return None
        raw_value = str(perturb_stage).strip()
        if raw_value == '':
            return None
        stage_order = self._stage_order()
        if raw_value in stage_order:
            return {
                'index': int(stage_order.index(raw_value)),
                'label': str(raw_value),
            }
        try:
            stage_index = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"perturb_stage must match one of the dataset stage labels: {', '.join(stage_order)}"
            ) from exc
        if stage_index < 0 or stage_index >= len(stage_order):
            raise ValueError(
                f"perturb_stage index {stage_index} is out of range for stages: {', '.join(stage_order)}"
            )
        return {
            'index': int(stage_index),
            'label': str(stage_order[stage_index]),
        }

    def _signature_hash(self, payload):
        encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.sha1(encoded).hexdigest()[:16]

    def _stage_signature(self, stage_name, **params):
        payload = {
            'version': PERTURBATION_STAGE_CACHE_VERSION,
            'stage': stage_name,
            'source': self._cache_source_identity(),
            'params': params,
        }
        payload['signature_hash'] = self._signature_hash(payload)
        return payload

    def _stage_cache_dir(self, stage_name):
        stage_dir = self._cache_root() / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _stage_manifest_path(self, stage_name):
        return self._stage_cache_dir(stage_name) / 'manifest.json'

    def _stage_payload_path(self, stage_name):
        return self._stage_cache_dir(stage_name) / 'payload.pkl'

    def _load_stage_cache(self, stage_name, signature):
        manifest_path = self._stage_manifest_path(stage_name)
        payload_path = self._stage_payload_path(stage_name)
        if not manifest_path.exists() or not payload_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('signature') != signature:
            return None
        with payload_path.open('rb') as handle:
            payload = pickle.load(handle)
        return {'payload': payload, 'manifest': manifest}

    def _save_stage_cache(self, stage_name, signature, payload, *, elapsed_seconds=None, metadata=None):
        payload_path = self._stage_payload_path(stage_name)
        manifest_path = self._stage_manifest_path(stage_name)
        with payload_path.open('wb') as handle:
            pickle.dump(payload, handle)
        manifest = {
            'stage': stage_name,
            'signature': signature,
            'payload_path': str(payload_path),
            'created_at': time.time(),
            'elapsed_seconds': elapsed_seconds,
            'metadata': metadata or {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest

    def _extract_uns_payload(self, keys, adata=None):
        target = self.adata if adata is None else adata
        return {
            key: copy.deepcopy(target.uns[key])
            for key in keys
            if key in target.uns
        }

    def _apply_uns_payload(self, payload, adata=None):
        target = self.adata if adata is None else adata
        for key, value in (payload or {}).items():
            target.uns[key] = copy.deepcopy(value)

    def _log_perturb_stage(self, logger, stage_name, *, status, started_at=None, **extra):
        parts = [f'perturb_stage={stage_name}', f'status={status}']
        if started_at is not None and status == 'DONE':
            parts.append(f'elapsed={ProgressLogger._format_duration(time.time() - started_at)}')
        for key, value in extra.items():
            parts.append(f'{key}={value}')
        logger.log(' '.join(parts))

    def _record_analysis_stage(self, stage_timings, stage_name, started_at, **extra):
        stage_timings[stage_name] = {
            'elapsed_seconds': time.time() - started_at,
            **extra,
        }

    def _write_analysis_stage_timings(self, stage_timings, *, analysis_started_at, metadata):
        payload = {
            'analysis_iteration': int(self.iteration),
            'target_dir': str(self.target_dir),
            'started_at': analysis_started_at,
            'completed_at': time.time(),
            'total_seconds': time.time() - analysis_started_at,
            'metadata': metadata,
            'stages': stage_timings,
        }
        timings_path = self.target_dir / 'analysis_stage_timings.json'
        timings_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload

    def _prepare_pathway_targets(self, *, logger, customized_pathway=None):
        signature = self._stage_signature(
            'pathway_target_prep',
            customized_pathway=self._path_fingerprint(customized_pathway) if customized_pathway is not None else None,
        )
        cached = self._load_stage_cache('pathway_target_prep', signature)

        logger.log('stage=pathway_overlap status=START')
        started_at = time.time()
        if cached is not None:
            payload = cached['payload']
            self._apply_uns_payload(payload)
            self._log_perturb_stage(logger, 'prepare_pathway_targets', status='DONE', started_at=started_at, cache='hit')
            logger.log('stage=pathway_overlap status=DONE')
            logger.log('stage=pathway_ranking status=START')
            logger.log('stage=pathway_ranking status=DONE')
            return payload, {'cache': 'hit', 'elapsed_seconds': time.time() - started_at}

        self._log_perturb_stage(logger, 'prepare_pathway_targets', status='START', cache='miss')
        if customized_pathway is not None:
            self.adata = calculateDataPathwayOverlapGene(self.adata,customized_pathway=customized_pathway)
        else:
            self.adata = calculateDataPathwayOverlapGene(self.adata)
        logger.log('stage=pathway_overlap status=DONE')
        logger.log('stage=pathway_ranking status=START')
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        logger.log('stage=pathway_ranking status=DONE')
        payload = self._extract_uns_payload(['data_pathway_overlap_genes', 'pathway_gene_len_dict', 'pathway_ranking'])
        self._save_stage_cache('pathway_target_prep', signature, payload, elapsed_seconds=time.time() - started_at)
        self._log_perturb_stage(logger, 'prepare_pathway_targets', status='DONE', started_at=started_at, cache='miss')
        return payload, {'cache': 'miss', 'elapsed_seconds': time.time() - started_at}

    def _prepare_drug_targets(self, *, logger):
        signature = self._stage_signature(
            'drug_target_prep',
            customized_drug=self._path_fingerprint(self.customized_drug) if self.customized_drug is not None else None,
            cmap_dir=self._path_fingerprint(self.cmap_dir) if self.cmap_dir is not None else None,
        )
        cached = self._load_stage_cache('drug_target_prep', signature)
        logger.log('stage=prepare_drug_targets status=START')
        started_at = time.time()
        if cached is not None:
            payload = cached['payload']
            self._apply_uns_payload(payload)
            logger.log('stage=prepare_drug_targets status=DONE')
            self._log_perturb_stage(logger, 'prepare_drug_targets', status='DONE', started_at=started_at, cache='hit')
            return payload, {'cache': 'hit', 'elapsed_seconds': time.time() - started_at}

        self._log_perturb_stage(logger, 'prepare_drug_targets', status='START', cache='miss')
        direction_flag = False
        customized_direction = None
        if self.customized_drug is not None:
            try:
                temp_drug = np.load(self.customized_drug,allow_pickle=True).item()
                if ':' in temp_drug[list(temp_drug.keys())[0]][0]:
                    direction_flag = True
            except Exception as exc:
                raise ValueError('Invalid customized drug format.') from exc
            if direction_flag:
                customized_direction = self.customized_drug
            self.adata = find_overlap_and_assign_direction(
                self.adata,
                customized_drug=self.customized_drug,
                customized_direction=customized_direction,
            )
        else:
            if self.cmap_dir is not None:
                self.adata = find_overlap_and_assign_direction(self.adata,cmap_dir=self.cmap_dir)
            else:
                raise ValueError('Please provide a cmap_dir or a customized drug database.')
        payload = self._extract_uns_payload(['data_drug_overlap_genes', 'drug-gene_len_dict'])
        self._save_stage_cache('drug_target_prep', signature, payload, elapsed_seconds=time.time() - started_at)
        logger.log('stage=prepare_drug_targets status=DONE')
        self._log_perturb_stage(logger, 'prepare_drug_targets', status='DONE', started_at=started_at, cache='miss')
        return payload, {'cache': 'miss', 'elapsed_seconds': time.time() - started_at}

    def _background_size_signature(self, payload):
        return sorted(len(value) for value in (payload or {}).values())

    def _build_cmap_random_background_genes(self, *, sample_count):
        random_gene = self.cmap_overlapped_genes(self.adata.uns['data_drug_overlap_genes'])
        random_genes = []
        import random

        for sample_idx in range(sample_count):
            random.seed(sample_idx)
            choices = [random.choice(['+','-']) for _ in range(len(random_gene))]
            random_genes.append([random_gene[i] + ':' + choices[i] for i in range(len(random_gene))])
        return random_genes

    def _clear_stage_keys(self, adata, keys):
        for key in keys:
            adata.uns.pop(key, None)

    def _ensure_perturbation_runner(self, *, execution_backend, logger, cache_state):
        runner_mode = 'exact' if execution_backend in {'gpu_exact', 'cpu_exact_parallel'} else 'centroid'
        cache_key = f'runner:{execution_backend}'
        runner = cache_state.get(cache_key)
        if runner is not None:
            return runner
        started_at = time.time()
        self._log_perturb_stage(logger, 'build_load_runner', status='START')
        if runner_mode == 'exact':
            runner_kwargs = {}
            if execution_backend == 'cpu_exact_parallel':
                runner_kwargs['device_override'] = 'cpu'
            elif execution_backend == 'gpu_exact':
                runner_kwargs['device_override'] = 'cuda:0'
            try:
                runner = perturbation(
                    self.adata,
                    self.target_dir/'model_save'/self.model_name,
                    self.target_dir/'idrem',
                    **runner_kwargs,
                )
            except TypeError as exc:
                # Test doubles and legacy runners may not accept the CPU override kwarg.
                if 'device_override' in str(exc):
                    runner = perturbation(
                        self.adata,
                        self.target_dir/'model_save'/self.model_name,
                        self.target_dir/'idrem',
                    )
                else:
                    raise
        else:
            runner = perturbation_centroid(self.adata, self.target_dir/'model_save'/self.model_name,self.target_dir/'idrem')
        cache_state[cache_key] = runner
        cache_state['runner'] = runner
        cache_state['runner_builds'] = cache_state.get('runner_builds', 0) + 1
        self._log_perturb_stage(
            logger,
            'build_load_runner',
            status='DONE',
            started_at=started_at,
            execution=execution_backend,
        )
        return runner

    def _run_or_load_pathway_task(
        self,
        *,
        logger,
        execution_backend,
        cpu_workers,
        perturb_change,
        direction_mode,
        perturbed_tracks,
        overall_perturbation_analysis,
        perturb_stage,
        cache_state,
    ):
        pathway_items = self.adata.uns.get('data_pathway_overlap_genes', {})
        include_grouped_tracks = self._should_include_grouped_track_scores(
            perturbed_tracks=perturbed_tracks,
            overall_perturbation_analysis=overall_perturbation_analysis,
            perturb_stage=perturb_stage,
        )
        background_signature = self._stage_signature(
            'pathway_background',
            schedule='single_gpu_optimized',
            execution=str(execution_backend),
            perturb_change=float(perturb_change),
            direction_mode=str(direction_mode),
            size_signature=self._background_size_signature(pathway_items),
            perturb_stage=perturb_stage['label'] if perturb_stage is not None else None,
        )
        result_signature = self._stage_signature(
            'pathway_result',
            schedule='single_gpu_optimized',
            execution=str(execution_backend),
            perturb_change=float(perturb_change),
            direction_mode=str(direction_mode),
            perturbed_tracks=self._normalize_track_request(perturbed_tracks),
            overall=bool(overall_perturbation_analysis),
            perturb_stage=perturb_stage['label'] if perturb_stage is not None else None,
            include_grouped_tracks=bool(include_grouped_tracks),
            background_signature=background_signature['signature_hash'],
        )

        logger.log('task=pathway_perturbation status=START')
        stage_metrics = {
            'background': {'cache': 'miss', 'elapsed_seconds': 0.0},
            'forward': {'cache': 'miss', 'elapsed_seconds': 0.0, 'backend': str(execution_backend)},
            'score': {'cache': 'miss', 'elapsed_seconds': 0.0},
        }
        result_cached = self._load_stage_cache('pathway_result', result_signature)
        if result_cached is not None:
            payload = result_cached['payload']
            self._apply_uns_payload(payload)
            bg_cached = self._load_stage_cache('pathway_background', background_signature)
            background_payload = bg_cached['payload'] if bg_cached is not None else {}
            if background_payload:
                self.adata.uns['pathway_random_background_perturbation_deltaD'] = copy.deepcopy(
                    background_payload.get('random_background_perturbation_deltaD', {})
                )
            self._log_perturb_stage(logger, 'pathway_background', status='DONE', cache='hit')
            self._log_perturb_stage(logger, 'pathway_forward', status='DONE', cache='hit')
            self._log_perturb_stage(logger, 'pathway_score', status='DONE', cache='hit')
            logger.log('task=pathway_perturbation status=DONE')
            stage_metrics['background']['cache'] = 'hit'
            stage_metrics['forward']['cache'] = 'hit'
            stage_metrics['score']['cache'] = 'hit'
            return payload, background_payload, stage_metrics

        runner = self._ensure_perturbation_runner(execution_backend=execution_backend, logger=logger, cache_state=cache_state)
        self._clear_stage_keys(runner.adata, ['random_background_perturbation_deltaD', 'pathway_perturbation_deltaD', 'pathway_perturbation_score'])
        use_cuda = str(execution_backend).startswith('gpu_')
        device = 'cuda:0' if use_cuda else 'cpu'

        background_cached = self._load_stage_cache('pathway_background', background_signature)
        if background_cached is not None:
            background_started_at = time.time()
            background_payload = background_cached['payload']
            runner.adata.uns['random_background_perturbation_deltaD'] = copy.deepcopy(
                background_payload.get('random_background_perturbation_deltaD', {})
            )
            self._log_perturb_stage(logger, 'pathway_background', status='DONE', cache='hit')
            stage_metrics['background'] = {
                'cache': 'hit',
                'elapsed_seconds': time.time() - background_started_at,
            }
        else:
            started_at = time.time()
            self._log_perturb_stage(logger, 'pathway_background', status='START', cache='miss')
            runner.run('random_pathway_background',perturb_change,inplace=True,CUDA=use_cuda,device=device,direction_mode=direction_mode)
            background_payload = {
                'random_background_perturbation_deltaD': copy.deepcopy(
                    runner.adata.uns.get('random_background_perturbation_deltaD', {})
                ),
            }
            self._save_stage_cache('pathway_background', background_signature, background_payload, elapsed_seconds=time.time() - started_at)
            self._log_perturb_stage(logger, 'pathway_background', status='DONE', started_at=started_at, cache='miss')
            stage_metrics['background'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
            }

        started_at = time.time()
        self._log_perturb_stage(logger, 'pathway_forward', status='START')
        if execution_backend == 'cpu_exact_parallel':
            payload = run_exact_parallel_delta(
                dataset_path=self.data_path,
                attribute_path=self.attribute_path,
                model_path=self.model_source_path,
                training_params_path=self.training_params_path,
                idrem_dir=self.target_dir/'idrem',
                backend_mode='pathway',
                items=runner.adata.uns.get('data_pathway_overlap_genes', {}),
                perturb_change=float(perturb_change),
                direction_mode=str(direction_mode),
                cpu_workers=cpu_workers,
            )
            runner.adata.uns['pathway_perturbation_deltaD'] = copy.deepcopy(payload)
            self._log_perturb_stage(logger, 'pathway_forward', status='DONE', started_at=started_at, backend=execution_backend, cpu_workers=cpu_workers)
            stage_metrics['forward'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
                'backend': str(execution_backend),
                'cpu_workers': int(cpu_workers),
            }
        else:
            runner.run('pathway',perturb_change,inplace=True,CUDA=use_cuda,device=device,direction_mode=direction_mode)
            self._log_perturb_stage(logger, 'pathway_forward', status='DONE', started_at=started_at, backend=execution_backend)
            stage_metrics['forward'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
                'backend': str(execution_backend),
            }

        started_at = time.time()
        self._log_perturb_stage(logger, 'pathway_score', status='START')
        if perturbed_tracks != 'all':
            runner.analysis(
                'pathway',
                perturb_change,
                perturbed_tracks,
                overall_perturbation_analysis=False,
                stage=perturb_stage['index'] if perturb_stage is not None else None,
                direction_mode=direction_mode,
            )
        else:
            runner.analysis(
                'pathway',
                perturb_change,
                perturbed_tracks,
                overall_perturbation_analysis,
                stage=perturb_stage['index'] if perturb_stage is not None else None,
                direction_mode=direction_mode,
            )
            if include_grouped_tracks:
                self._augment_score_bucket_with_grouped_tracks(
                    runner,
                    mode='pathway',
                    perturb_change=perturb_change,
                    direction_mode=direction_mode,
                    stage=perturb_stage['index'] if perturb_stage is not None else None,
                )
        self._log_perturb_stage(logger, 'pathway_score', status='DONE', started_at=started_at)
        stage_metrics['score'] = {
            'cache': 'miss',
            'elapsed_seconds': time.time() - started_at,
        }

        payload = self._extract_uns_payload(['pathway_perturbation_deltaD', 'pathway_perturbation_score'], adata=runner.adata)
        self._save_stage_cache(
            'pathway_result',
            result_signature,
            payload,
            metadata={'background_signature': background_signature['signature_hash']},
        )
        runner.adata.uns['pathway_random_background_perturbation_deltaD'] = copy.deepcopy(
            background_payload.get('random_background_perturbation_deltaD', {})
        )
        logger.log('task=pathway_perturbation status=DONE')
        return payload, background_payload, stage_metrics

    def _run_or_load_drug_task(
        self,
        *,
        logger,
        execution_backend,
        cpu_workers,
        perturb_change,
        direction_mode,
        perturbed_tracks,
        overall_perturbation_analysis,
        perturb_stage,
        random_times,
        progressionmarker_background_sampling,
        cache_state,
    ):
        drug_items = self.adata.uns.get('data_drug_overlap_genes', {})
        include_grouped_tracks = self._should_include_grouped_track_scores(
            perturbed_tracks=perturbed_tracks,
            overall_perturbation_analysis=overall_perturbation_analysis,
            perturb_stage=perturb_stage,
        )
        if self.customized_drug is not None:
            background_size_signature = [int(self.get_median_random_gene(drug_items))]
        else:
            background_size_signature = [len(self.cmap_overlapped_genes(drug_items))]
        background_signature = self._stage_signature(
            'drug_background',
            schedule='single_gpu_optimized',
            execution=str(execution_backend),
            perturb_change=float(perturb_change),
            direction_mode=str(direction_mode),
            size_signature=background_size_signature,
            random_times=int(random_times),
            customized_drug=bool(self.customized_drug is not None),
            perturb_stage=perturb_stage['label'] if perturb_stage is not None else None,
        )
        result_signature = self._stage_signature(
            'drug_result',
            schedule='single_gpu_optimized',
            execution=str(execution_backend),
            perturb_change=float(perturb_change),
            direction_mode=str(direction_mode),
            perturbed_tracks=self._normalize_track_request(perturbed_tracks),
            overall=bool(overall_perturbation_analysis),
            perturb_stage=perturb_stage['label'] if perturb_stage is not None else None,
            include_grouped_tracks=bool(include_grouped_tracks),
            background_signature=background_signature['signature_hash'],
        )

        logger.log('task=drug_perturbation status=START')
        stage_metrics = {
            'background': {'cache': 'miss', 'elapsed_seconds': 0.0},
            'forward': {'cache': 'miss', 'elapsed_seconds': 0.0, 'backend': str(execution_backend)},
            'score': {'cache': 'miss', 'elapsed_seconds': 0.0},
        }
        result_cached = self._load_stage_cache('drug_result', result_signature)
        if result_cached is not None:
            payload = result_cached['payload']
            self._apply_uns_payload(payload)
            bg_cached = self._load_stage_cache('drug_background', background_signature)
            background_payload = bg_cached['payload'] if bg_cached is not None else {}
            if background_payload:
                self.adata.uns['drug_random_background_perturbation_deltaD'] = copy.deepcopy(
                    background_payload.get('random_background_perturbation_deltaD', {})
                )
            self._log_perturb_stage(logger, 'drug_background', status='DONE', cache='hit')
            self._log_perturb_stage(logger, 'drug_forward', status='DONE', cache='hit')
            self._log_perturb_stage(logger, 'drug_score', status='DONE', cache='hit')
            logger.log('task=drug_perturbation status=DONE')
            stage_metrics['background']['cache'] = 'hit'
            stage_metrics['forward']['cache'] = 'hit'
            stage_metrics['score']['cache'] = 'hit'
            return payload, background_payload, stage_metrics

        runner = self._ensure_perturbation_runner(execution_backend=execution_backend, logger=logger, cache_state=cache_state)
        self._clear_stage_keys(runner.adata, ['random_background_perturbation_deltaD', 'drug_perturbation_deltaD', 'drug_perturbation_score'])
        use_cuda = str(execution_backend).startswith('gpu_')
        device = 'cuda:0' if use_cuda else 'cpu'

        background_cached = self._load_stage_cache('drug_background', background_signature)
        if background_cached is not None:
            background_started_at = time.time()
            background_payload = background_cached['payload']
            runner.adata.uns['random_background_perturbation_deltaD'] = copy.deepcopy(
                background_payload.get('random_background_perturbation_deltaD', {})
            )
            self._log_perturb_stage(logger, 'drug_background', status='DONE', cache='hit')
            stage_metrics['background'] = {
                'cache': 'hit',
                'elapsed_seconds': time.time() - background_started_at,
            }
        else:
            started_at = time.time()
            self._log_perturb_stage(logger, 'drug_background', status='START', cache='miss')
            if self.customized_drug is not None:
                runner.run(
                    'random_drug_background',
                    perturb_change,
                    inplace=True,
                    random_times=random_times,
                    random_genes=self.get_median_random_gene(runner.adata.uns['data_drug_overlap_genes']),
                    CUDA=use_cuda,
                    device=device,
                    direction_mode=direction_mode,
                )
            else:
                random_genes = self._build_cmap_random_background_genes(
                    sample_count=min(
                        len(runner.adata.uns.get('data_drug_overlap_genes', {})),
                        int(random_times),
                    )
                )
                runner.adata.uns['cmap_random_genes'] = random_genes
                runner.run(
                    'random_drug_background',
                    perturb_change,
                    inplace=True,
                    random_times=len(random_genes),
                    random_genes=random_genes,
                    CUDA=use_cuda,
                    device=device,
                    direction_mode=direction_mode,
                )
            background_payload = {
                'random_background_perturbation_deltaD': copy.deepcopy(
                    runner.adata.uns.get('random_background_perturbation_deltaD', {})
                ),
            }
            self._save_stage_cache('drug_background', background_signature, background_payload, elapsed_seconds=time.time() - started_at)
            self._log_perturb_stage(logger, 'drug_background', status='DONE', started_at=started_at, cache='miss')
            stage_metrics['background'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
            }

        started_at = time.time()
        self._log_perturb_stage(logger, 'drug_forward', status='START')
        if execution_backend == 'cpu_exact_parallel':
            payload = run_exact_parallel_delta(
                dataset_path=self.data_path,
                attribute_path=self.attribute_path,
                model_path=self.model_source_path,
                training_params_path=self.training_params_path,
                idrem_dir=self.target_dir/'idrem',
                backend_mode='drug',
                items=runner.adata.uns.get('data_drug_overlap_genes', {}),
                perturb_change=float(perturb_change),
                direction_mode=str(direction_mode),
                cpu_workers=cpu_workers,
            )
            runner.adata.uns['drug_perturbation_deltaD'] = copy.deepcopy(payload)
            self._log_perturb_stage(logger, 'drug_forward', status='DONE', started_at=started_at, backend=execution_backend, cpu_workers=cpu_workers)
            stage_metrics['forward'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
                'backend': str(execution_backend),
                'cpu_workers': int(cpu_workers),
            }
        else:
            runner.run('drug',perturb_change,inplace=True,CUDA=use_cuda,device=device,direction_mode=direction_mode)
            self._log_perturb_stage(logger, 'drug_forward', status='DONE', started_at=started_at, backend=execution_backend)
            stage_metrics['forward'] = {
                'cache': 'miss',
                'elapsed_seconds': time.time() - started_at,
                'backend': str(execution_backend),
            }

        started_at = time.time()
        self._log_perturb_stage(logger, 'drug_score', status='START')
        if perturbed_tracks!='all':
            runner.analysis(
                'drug',
                perturb_change,
                perturbed_tracks,
                overall_perturbation_analysis=False,
                stage=perturb_stage['index'] if perturb_stage is not None else None,
                direction_mode=direction_mode,
            )
        else:
            runner.analysis(
                'drug',
                perturb_change,
                perturbed_tracks,
                overall_perturbation_analysis,
                stage=perturb_stage['index'] if perturb_stage is not None else None,
                direction_mode=direction_mode,
            )
            if include_grouped_tracks:
                self._augment_score_bucket_with_grouped_tracks(
                    runner,
                    mode='drug',
                    perturb_change=perturb_change,
                    direction_mode=direction_mode,
                    stage=perturb_stage['index'] if perturb_stage is not None else None,
                )
        self._log_perturb_stage(logger, 'drug_score', status='DONE', started_at=started_at)
        stage_metrics['score'] = {
            'cache': 'miss',
            'elapsed_seconds': time.time() - started_at,
        }

        payload = self._extract_uns_payload(['drug_perturbation_deltaD', 'drug_perturbation_score'], adata=runner.adata)
        self._save_stage_cache(
            'drug_result',
            result_signature,
            payload,
            metadata={'background_signature': background_signature['signature_hash']},
        )
        runner.adata.uns['drug_random_background_perturbation_deltaD'] = copy.deepcopy(
            background_payload.get('random_background_perturbation_deltaD', {})
        )
        logger.log('task=drug_perturbation status=DONE')
        return payload, background_payload, stage_metrics

    def _copy_single_gene_background(self, adata, bound, direction_mode):
        resolved_mode = infer_direction_mode(
            adata.uns['single_gene_perturbation_deltaD'],
            bound,
            direction_mode,
        )
        payload = {
            str(bound): adata.uns['single_gene_perturbation_deltaD'][str(bound)],
        }
        if resolved_mode == 'two_sided':
            payload[str(1 / bound)] = adata.uns['single_gene_perturbation_deltaD'][str(1 / bound)]
        adata.uns['random_background_perturbation_deltaD'] = payload
        self._record_background_metadata(adata, bound=bound, direction_mode=resolved_mode)
        return resolved_mode

    def _record_background_metadata(self, adata, *, bound, direction_mode):
        metadata = adata.uns.get('analysis_metadata')
        if not isinstance(metadata, dict):
            metadata = {}
        metadata['direction_mode'] = str(direction_mode)
        metadata['background_direction_mode'] = str(direction_mode)
        metadata['background_perturb_change'] = float(bound)
        adata.uns['analysis_metadata'] = metadata
        adata.uns['analysis_direction_mode'] = str(direction_mode)
        adata.uns['perturbation_background_direction_mode'] = str(direction_mode)
        adata.uns['perturbation_background_perturb_change'] = float(bound)
        adata.uns['random_background_direction_mode'] = str(direction_mode)
        adata.uns['random_background_perturb_change'] = float(bound)

    def _should_include_grouped_track_scores(
        self,
        *,
        perturbed_tracks,
        overall_perturbation_analysis,
        perturb_stage,
    ):
        return (
            perturbed_tracks == 'all'
            and bool(overall_perturbation_analysis)
            and perturb_stage is None
        )

    def _sorted_track_keys(self, values):
        def sort_key(value):
            text = str(value)
            if text == 'overall':
                return (-1, text)
            pieces = []
            for part in text.split('-'):
                pieces.append(int(part) if part.isdigit() else part)
            return (0, pieces)

        return [str(value) for value in sorted(values, key=sort_key)]

    def _grouped_track_heads(self, delta_payload, perturb_change):
        perturb_key = str(perturb_change)
        bucket = delta_payload.get(perturb_key, {}) if isinstance(delta_payload, dict) else {}
        grouped = []
        seen = set()
        for raw_track in self._sorted_track_keys(bucket.keys()):
            if str(raw_track) == 'overall':
                continue
            head = str(raw_track).split('-')[0].strip()
            if not head or head in seen:
                continue
            seen.add(head)
            grouped.append(head)
        return grouped

    def _augment_score_bucket_with_grouped_tracks(
        self,
        perturbation_runner,
        *,
        mode,
        perturb_change,
        direction_mode,
        stage=None,
    ):
        score_key = f'{mode}_perturbation_score'
        existing_payload = perturbation_runner.adata.uns.get(score_key, {})
        resolved_mode = infer_direction_mode(existing_payload, perturb_change, direction_mode)
        perturb_key = str(perturb_change)
        overall_bucket = copy.deepcopy(existing_payload.get(perturb_key, {}).get('overall'))
        delta_key = f'{mode}_perturbation_deltaD'
        grouped_tracks = self._grouped_track_heads(
            perturbation_runner.adata.uns.get(delta_key, {}),
            perturb_change,
        )
        track_bucket = copy.deepcopy(
            perturbation_runner.adata.uns.get(score_key, {}).get(perturb_key, {})
        )
        merged = {}
        if overall_bucket is not None:
            merged['overall'] = overall_bucket
        for track_group in grouped_tracks:
            perturbation_runner.analysis(
                mode,
                perturb_change,
                [str(track_group)],
                overall_perturbation_analysis=True,
                stage=stage,
                direction_mode=resolved_mode,
            )
            group_bucket = (
                perturbation_runner.adata.uns.get(score_key, {})
                .get(perturb_key, {})
                .get('overall')
            )
            if group_bucket is None:
                continue
            merged[str(track_group)] = copy.deepcopy(group_bucket)
        self._rebuild_directional_score_bucket(
            perturbation_runner,
            score_key,
            perturb_change,
            merged,
            resolved_mode,
        )
        return merged

    def _rebuild_directional_score_bucket(self, perturbation_runner, score_key, perturb_change, track_results, direction_mode):
        perturbation_runner.adata.uns.pop(score_key, None)
        perturbation_runner.adata.uns[score_key] = {}
        store_directional_result(
            perturbation_runner.adata.uns[score_key],
            perturb_change,
            track_results,
            direction_mode=direction_mode,
        )
        
    def start_analyse(self,progressionmarker_background_sampling,run_perturbation,random_times, ignore_dynamic_markers=False, ignore_hcmarkers=False,customized_pathway=None,perturb_change=0.5,overall_perturbation_analysis=True,perturbed_tracks='all',perturb_stage=None,ignore_pathway_perturabtion=False,ignore_drug_perturabtion=False,centroid=False,direction_mode='two_sided',perturbation_schedule='single_gpu_optimized',cpu_workers=None,dynamic_marker_strategy='optimized',dynamic_marker_cpu_workers=None,dynamic_marker_force_serial=False,dynamic_marker_verify_legacy=False,hierarchical_marker_cpu_workers=None):
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
        
        perturbation_schedule = self._normalize_perturbation_schedule(perturbation_schedule)
        dynamic_marker_strategy = self._normalize_dynamic_marker_strategy(dynamic_marker_strategy)
        logger = ProgressLogger(prefix='[UNAGI-ANALYSIS]')
        analysis_t0 = time.time()
        logger.log(
            f"start analyse iteration={self.iteration}, run_perturbation={run_perturbation}, "
            f"centroid={centroid}, perturb_change={perturb_change}, direction_mode={direction_mode}, "
            f"perturbation_schedule={perturbation_schedule}, dynamic_marker_strategy={dynamic_marker_strategy}"
        )
        perturb_stage_info = self._resolve_perturb_stage(perturb_stage)
        execution_context = self._resolve_perturb_execution_backend(centroid=centroid, cpu_workers=cpu_workers)
        stage_timings = {}
        hierarchical_workers = (
            int(hierarchical_marker_cpu_workers)
            if hierarchical_marker_cpu_workers not in (None, '')
            else 1
        )
        logger.log(
            f"perturbation_backend={execution_context['backend']} "
            f"cpu_workers={execution_context['cpu_workers']} device={execution_context['device']} "
            f"perturb_stage={perturb_stage_info['label'] if perturb_stage_info is not None else 'all'}"
        )

        def _fallback_to_cpu_if_needed(exc, *, task_name):
            if not self._is_gpu_execution_backend(execution_context['backend']):
                raise exc
            if not self._is_gpu_runtime_error(exc):
                raise exc
            fallback_backend = self._cpu_fallback_backend(execution_context['backend'])
            logger.log(
                f"task={task_name} status=GPU_FALLBACK "
                f"from_backend={execution_context['backend']} "
                f"to_backend={fallback_backend} reason={type(exc).__name__}"
            )
            cache_state_key = f"runner:{execution_context['backend']}"
            perturbation_state.pop(cache_state_key, None)
            perturbation_state.pop('runner', None)
            gc.collect()
            execution_context['backend'] = fallback_backend
            execution_context['use_cuda'] = False
            execution_context['device'] = 'cpu'
            logger.log(
                f"perturbation_backend={execution_context['backend']} "
                f"cpu_workers={execution_context['cpu_workers']} device={execution_context['device']}"
            )

        if not ignore_hcmarkers:
            logger.log('stage=hierarchical_markers status=START')
            stage_started_at = time.time()
            hcmarkers= get_dataset_hcmarkers(
                self.adata,
                stage_key='stage',
                cluster_key='leiden',
                use_rep='umaps',
                strategy='optimized',
                cpu_workers=hierarchical_workers,
            )
            logger.log('stage=hierarchical_markers status=DONE')
            self._record_analysis_stage(
                stage_timings,
                'hierarchical_markers',
                stage_started_at,
                strategy='optimized',
                cpu_workers=int(hierarchical_workers),
            )

        stage_started_at = time.time()
        _pathway_targets_payload, pathway_prep_meta = self._prepare_pathway_targets(logger=logger, customized_pathway=customized_pathway)
        self._record_analysis_stage(
            stage_timings,
            'pathway_target_prep',
            stage_started_at,
            cache=pathway_prep_meta['cache'],
        )
        if not os.path.exists(self.target_dir/'idrem'):
            shutil.copytree(self.data_folder.parent/'idremResults', self.target_dir/'idrem')
        
        (self.target_dir/'model_save').mkdir(parents=True, exist_ok=True)
        shutil.copy(self.data_folder.parent.parent/'model_save'/self.model_name, self.target_dir/'model_save'/self.model_name)
        shutil.copy(self.data_folder.parent.parent/'model_save'/'training_parameters.json', self.target_dir/'model_save'/'training_parameters.json')
        logger.log('stage=copy_model_artifacts status=DONE')
        # stop 10 seconds to ensure the model is copied
        time.sleep(10)
        if not ignore_dynamic_markers:
            logger.log('stage=progression_markers status=START')
            dynamic_marker_cache_dir = self.target_dir / 'dynamic_marker_cache'
            dynamic_marker_workers = (
                int(dynamic_marker_cpu_workers)
                if dynamic_marker_cpu_workers not in (None, '')
                else default_cpu_workers(None)
            )
            background_path = self.target_dir / (str(progressionmarker_background_sampling) + 'progressionmarker_background.npy')
            idrem_results_dir = os.path.join(os.path.dirname(self.data_folder), 'idremResults')

            if os.path.exists(background_path):
                background_started_at = time.time()
                logger.log('progression_background source=cached')
                progressionmarker_background = np.load(background_path,allow_pickle=True)
                progressionmarker_background = dict(progressionmarker_background.tolist())
                self._record_analysis_stage(
                    stage_timings,
                    'dynamic_markers_background',
                    background_started_at,
                    cache='hit',
                    strategy=str(dynamic_marker_strategy),
                    cpu_workers=int(dynamic_marker_workers),
                )
            else:
                logger.log(
                    f"progression_background source=build times={progressionmarker_background_sampling} "
                    f"strategy={dynamic_marker_strategy} cpu_workers={dynamic_marker_workers} "
                    f"force_serial={bool(dynamic_marker_force_serial)}"
                )
                background_started_at = time.time()
                progressionmarker_background = get_progressionmarker_background(
                    times=progressionmarker_background_sampling,
                    adata=self.adata,
                    total_stage=self.total_stage,
                    strategy=dynamic_marker_strategy,
                    dataset_path=self.data_path,
                    attribute_path=self.attribute_path,
                    cache_dir=dynamic_marker_cache_dir,
                    cpu_workers=dynamic_marker_workers,
                    force_serial=dynamic_marker_force_serial,
                )
                np.save(background_path,progressionmarker_background)
                self._record_analysis_stage(
                    stage_timings,
                    'dynamic_markers_background',
                    background_started_at,
                    cache='miss',
                    strategy=str(dynamic_marker_strategy),
                    cpu_workers=int(dynamic_marker_workers),
                )
            scoring_started_at = time.time()
            self.adata.uns['progressionMarkers'] = runGetProgressionMarker_one_dist(
                idrem_results_dir,
                progressionmarker_background,
                self.adata.shape[1],
                cutoff=0.05,
                strategy=dynamic_marker_strategy,
            )
            self._record_analysis_stage(
                stage_timings,
                'dynamic_markers_scoring',
                scoring_started_at,
                strategy=str(dynamic_marker_strategy),
            )
            if dynamic_marker_verify_legacy and dynamic_marker_strategy != 'legacy':
                from .marker_discovery.dynamic_markers_helper import compare_progressionmarker_strategies

                comparison = compare_progressionmarker_strategies(
                    times=progressionmarker_background_sampling,
                    adata=self.adata,
                    total_stage=self.total_stage,
                    idrem_directory=idrem_results_dir,
                    size=self.adata.shape[1],
                    cutoff=0.05,
                    dataset_path=self.data_path,
                    attribute_path=self.attribute_path,
                    cache_dir=dynamic_marker_cache_dir,
                    cpu_workers=dynamic_marker_workers,
                )
                if comparison['ok']:
                    logger.log('dynamic_marker_parity status=PASS')
                else:
                    logger.log(f"dynamic_marker_parity status=FAIL detail={comparison['comparisons']}")
                    progressionmarker_background = comparison['legacy_background']
                    self.adata.uns['progressionMarkers'] = comparison['legacy_markers']
                    np.save(background_path, progressionmarker_background)
            logger.log('stage=progression_markers status=DONE')
        perturbation_state = {'runner': None, 'runner_builds': 0}
        if run_perturbation:
            logger.log('stage=perturbation status=START')
            pathway_background_payload = None
            drug_background_payload = None
            if not ignore_drug_perturabtion:
                stage_started_at = time.time()
                _drug_targets_payload, drug_prep_meta = self._prepare_drug_targets(logger=logger)
                self._record_analysis_stage(
                    stage_timings,
                    'drug_target_prep',
                    stage_started_at,
                    cache=drug_prep_meta['cache'],
                )
            logger.log('stage=perturbation_compute status=START')
            gc.collect()
            if not ignore_pathway_perturabtion:
                try:
                    _pathway_payload, pathway_background_payload, pathway_stage_metrics = self._run_or_load_pathway_task(
                        logger=logger,
                        execution_backend=execution_context['backend'],
                        cpu_workers=execution_context['cpu_workers'],
                        perturb_change=perturb_change,
                        direction_mode=direction_mode,
                        perturbed_tracks=perturbed_tracks,
                        overall_perturbation_analysis=overall_perturbation_analysis,
                        perturb_stage=perturb_stage_info,
                        cache_state=perturbation_state,
                    )
                except Exception as exc:
                    _fallback_to_cpu_if_needed(exc, task_name='pathway_perturbation')
                    _pathway_payload, pathway_background_payload, pathway_stage_metrics = self._run_or_load_pathway_task(
                        logger=logger,
                        execution_backend=execution_context['backend'],
                        cpu_workers=execution_context['cpu_workers'],
                        perturb_change=perturb_change,
                        direction_mode=direction_mode,
                        perturbed_tracks=perturbed_tracks,
                        overall_perturbation_analysis=overall_perturbation_analysis,
                        perturb_stage=perturb_stage_info,
                        cache_state=perturbation_state,
                    )
                stage_timings['pathway_background'] = pathway_stage_metrics['background']
                stage_timings['pathway_forward'] = pathway_stage_metrics['forward']
                stage_timings['pathway_scoring'] = pathway_stage_metrics['score']
            else:
                logger.log('task=pathway_perturbation status=SKIP')
            if not ignore_drug_perturabtion:
                try:
                    _drug_payload, drug_background_payload, drug_stage_metrics = self._run_or_load_drug_task(
                        logger=logger,
                        execution_backend=execution_context['backend'],
                        cpu_workers=execution_context['cpu_workers'],
                        perturb_change=perturb_change,
                        direction_mode=direction_mode,
                        perturbed_tracks=perturbed_tracks,
                        overall_perturbation_analysis=overall_perturbation_analysis,
                        perturb_stage=perturb_stage_info,
                        random_times=random_times,
                        progressionmarker_background_sampling=progressionmarker_background_sampling,
                        cache_state=perturbation_state,
                    )
                except Exception as exc:
                    _fallback_to_cpu_if_needed(exc, task_name='drug_perturbation')
                    _drug_payload, drug_background_payload, drug_stage_metrics = self._run_or_load_drug_task(
                        logger=logger,
                        execution_backend=execution_context['backend'],
                        cpu_workers=execution_context['cpu_workers'],
                        perturb_change=perturb_change,
                        direction_mode=direction_mode,
                        perturbed_tracks=perturbed_tracks,
                        overall_perturbation_analysis=overall_perturbation_analysis,
                        perturb_stage=perturb_stage_info,
                        random_times=random_times,
                        progressionmarker_background_sampling=progressionmarker_background_sampling,
                        cache_state=perturbation_state,
                    )
                stage_timings['drug_background'] = drug_stage_metrics['background']
                stage_timings['drug_forward'] = drug_stage_metrics['forward']
                stage_timings['drug_scoring'] = drug_stage_metrics['score']
            else:
                logger.log('task=drug_perturbation status=SKIP')
            logger.log('stage=perturbation status=DONE')
            output_adata = perturbation_state['runner'].adata if perturbation_state['runner'] is not None else self.adata
            if pathway_background_payload is not None:
                output_adata.uns['pathway_random_background_perturbation_deltaD'] = copy.deepcopy(
                    pathway_background_payload.get('random_background_perturbation_deltaD', {})
                )
            if drug_background_payload is not None:
                output_adata.uns['drug_random_background_perturbation_deltaD'] = copy.deepcopy(
                    drug_background_payload.get('random_background_perturbation_deltaD', {})
                )
            compatibility_background = drug_background_payload or pathway_background_payload
            if compatibility_background is not None:
                output_adata.uns['random_background_perturbation_deltaD'] = copy.deepcopy(
                    compatibility_background.get('random_background_perturbation_deltaD', {})
                )
            self._record_background_metadata(output_adata, bound=perturb_change, direction_mode=direction_mode)
        else:
            output_adata = self.adata
            metadata = output_adata.uns.get('analysis_metadata')
            if not isinstance(metadata, dict):
                metadata = {}
            metadata['direction_mode'] = str(direction_mode)
            metadata['perturbation_schedule'] = str(perturbation_schedule)
            metadata['resolved_execution_backend'] = str(execution_context['backend'])
            metadata['cpu_workers'] = int(execution_context['cpu_workers'])
            metadata['perturb_stage'] = perturb_stage_info['label'] if perturb_stage_info is not None else None
            output_adata.uns['analysis_metadata'] = metadata
            output_adata.uns['analysis_direction_mode'] = str(direction_mode)
        if perturbation_state['runner'] is not None:
            output_adata = perturbation_state['runner'].adata
        else:
            output_adata = self.adata
        metadata = output_adata.uns.get('analysis_metadata')
        if not isinstance(metadata, dict):
            metadata = {}
        metadata['perturbation_schedule'] = str(perturbation_schedule)
        metadata['resolved_execution_backend'] = str(execution_context['backend'])
        metadata['cpu_workers'] = int(execution_context['cpu_workers'])
        metadata['perturbation_runner_builds'] = int(perturbation_state.get('runner_builds', 0))
        metadata['dynamic_marker_strategy'] = str(dynamic_marker_strategy)
        metadata['dynamic_marker_cpu_workers'] = int(dynamic_marker_workers) if not ignore_dynamic_markers else int(default_cpu_workers(None))
        metadata['dynamic_marker_force_serial'] = bool(dynamic_marker_force_serial)
        metadata['dynamic_marker_verify_legacy'] = bool(dynamic_marker_verify_legacy)
        metadata['hierarchical_marker_cpu_workers'] = int(hierarchical_workers)
        metadata['perturb_stage'] = perturb_stage_info['label'] if perturb_stage_info is not None else None
        output_adata.uns['analysis_metadata'] = metadata
        if not ignore_hcmarkers:
            output_adata.uns['hcmarkers'] = hcmarkers 
        logger.log('stage=save_outputs status=START')
        save_started_at = time.time()
        with open(os.path.join(self.target_dir,'attribute.pkl'),'wb') as f:
            pickle.dump(output_adata.uns,f)
        del output_adata.uns
        output_adata.obs['leiden'] = output_adata.obs['leiden'].astype(str)
        output_adata.obs['stage'] = output_adata.obs['stage'].astype(str)
        output_adata.obs['ident'] = output_adata.obs['ident'].astype(str)
        dataset_path = Path(self.target_dir) / 'dataset.h5ad'
        temp_dataset_path = Path(self.target_dir) / f".dataset.{os.getpid()}.{int(time.time())}.h5ad"
        # Write to a unique temporary file first so readers of an older
        # dataset.h5ad do not block the final analysis save on HDF5 locks.
        output_adata.write(temp_dataset_path, compression='gzip', compression_opts=9)
        os.replace(temp_dataset_path, dataset_path)
        self._record_analysis_stage(stage_timings, 'save_merge', save_started_at)
        self._write_analysis_stage_timings(
            stage_timings,
            analysis_started_at=analysis_t0,
            metadata={
                'resolved_execution_backend': str(execution_context['backend']),
                'cpu_workers': int(execution_context['cpu_workers']),
                'perturbation_schedule': str(perturbation_schedule),
                'dynamic_marker_strategy': str(dynamic_marker_strategy),
                'dynamic_marker_cpu_workers': int(dynamic_marker_workers) if not ignore_dynamic_markers else int(default_cpu_workers(None)),
                'hierarchical_marker_cpu_workers': int(hierarchical_workers),
                'run_perturbation': bool(run_perturbation),
                'perturb_stage': perturb_stage_info['label'] if perturb_stage_info is not None else None,
            },
        )
        logger.log(f"stage=save_outputs status=DONE elapsed={ProgressLogger._format_duration(time.time()-analysis_t0)}")
        # return a.adata
    def drug_perturbation_analysis(self,tracks,perturb_change, centroid=False, direction_mode=None):
        key = perturb_change
        track_results = {}
        resolved_mode = infer_direction_mode(self.adata.uns['drug_perturbation_deltaD'], key, direction_mode)
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
                perturbation_runner.analysis('drug',key,each_track,overall_perturbation_analysis=True,direction_mode=resolved_mode)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'drug_perturbation_score', key, track_results, resolved_mode)
        else:
            track_results = {}
            if  'overall' in self.adata.uns['drug_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('drug',key, tracks,overall_perturbation_analysis=True,direction_mode=resolved_mode)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['drug_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'drug_perturbation_score', key, track_results, resolved_mode)
        return perturbation_runner.adata
    def pathway_perturbation_analysis(self,tracks,perturb_change, centroid=False, direction_mode=None):
        key = perturb_change
        track_results = {}
        resolved_mode = infer_direction_mode(self.adata.uns['pathway_perturbation_deltaD'], key, direction_mode)
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
                perturbation_runner.analysis('pathway',key,each_track,overall_perturbation_analysis=True,direction_mode=resolved_mode)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'pathway_perturbation_score', key, track_results, resolved_mode)
        else:
            track_results = {}
            if  'overall' in self.adata.uns['pathway_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('pathway',key, tracks,overall_perturbation_analysis=True,direction_mode=resolved_mode)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['pathway_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'pathway_perturbation_score', key, track_results, resolved_mode)
        return perturbation_runner.adata
    def single_gene_perturbation_analysis(self,tracks,perturb_change, centroid=False, direction_mode=None):
        key = perturb_change
        track_results = {}
        resolved_mode = infer_direction_mode(self.adata.uns['single_gene_perturbation_deltaD'], key, direction_mode)
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
                perturbation_runner.analysis('single_gene',key,each_track,overall_perturbation_analysis=True,direction_mode=resolved_mode)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'single_gene_perturbation_score', key, track_results, resolved_mode)
        else:
            track_results = {}
            if  'overall' in self.adata.uns['single_gene_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('single_gene',key, tracks,overall_perturbation_analysis=True,direction_mode=resolved_mode)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['single_gene_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'single_gene_perturbation_score', key, track_results, resolved_mode)
        return perturbation_runner.adata
    def gene_combinatorial_perturbation_analysis(self,tracks,perturb_change, centroid=False, direction_mode=None):
        key = perturb_change
        track_results = {}
        resolved_mode = infer_direction_mode(self.adata.uns['two_genes_perturbation_deltaD'], key, direction_mode)
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
                perturbation_runner.analysis('two_genes',key,each_track,overall_perturbation_analysis=True,direction_mode=resolved_mode)
                track_results['track_'+each_track] = perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'two_genes_perturbation_score', key, track_results, resolved_mode)
        else:
            track_results = {}
            if  'overall' in self.adata.uns['two_genes_perturbation_score'][str(key)].keys():
                track_results['overall'] = self.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            perturbation_runner.analysis('two_genes',key, tracks,overall_perturbation_analysis=True,direction_mode=resolved_mode)
            track_results['track_'+tracks] = perturbation_runner.adata.uns['two_genes_perturbation_score'][str(key)]['overall'].copy()
            self._rebuild_directional_score_bucket(perturbation_runner, 'two_genes_perturbation_score', key, track_results, resolved_mode)
        return perturbation_runner.adata
