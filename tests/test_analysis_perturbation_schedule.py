import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import UNAGI.UNAGI_analyst as analyst_module
from UNAGI.perturbations.analysis_perturbation import _bh_adjust, perturbationAnalysis


class _FakePerturbationRunner:
    init_count = 0
    run_modes = []
    run_call_details = []
    analysis_modes = []

    def __init__(self, adata, model_name, idrem_dir):
        type(self).init_count += 1
        self.adata = adata

    @classmethod
    def reset(cls):
        cls.init_count = 0
        cls.run_modes = []
        cls.run_call_details = []
        cls.analysis_modes = []

    @staticmethod
    def _label_column_for_mode(mode):
        return {
            'pathway': 'compound',
            'drug': 'compound',
            'single_gene': 'gene',
            'two_genes': 'gene_combination',
            'two_drugs': 'drug_combination',
        }[mode]

    def _labels_for_mode(self, mode):
        if mode == 'pathway':
            return list(self.adata.uns.get('data_pathway_overlap_genes', {}).keys()) or ['PW1']
        if mode == 'drug':
            return list(self.adata.uns.get('data_drug_overlap_genes', {}).keys()) or ['DR1']
        if mode == 'single_gene':
            return [str(gene) for gene in self.adata.var_names.tolist()]
        if mode == 'two_genes':
            seeds = [str(gene) for gene in self.adata.uns.get('combinatorial_perturbation_genes_set1', [])]
            partners = [str(gene) for gene in self.adata.var_names.tolist()]
            return [f'{seed}_{partner}' for seed in seeds for partner in partners if seed != partner]
        if mode == 'two_drugs':
            return list(self.adata.uns.get('data_two_drug_overlap_genes', {}).keys()) or ['DR1__DR2']
        raise ValueError(f'Unsupported mode: {mode}')

    @staticmethod
    def _normalize_background_labels(random_genes):
        labels = []
        for index, item in enumerate(random_genes or []):
            if isinstance(item, (list, tuple, np.ndarray)):
                label = '__'.join(str(value) for value in item)
            else:
                label = str(item)
            labels.append(label if label else f'BG{index}')
        return labels or ['BG0']

    def _store_delta_payload(self, uns_key, bound, labels):
        self.adata.uns[uns_key] = {
            str(bound): {
                '0-1': {
                    str(label): np.array([float(index + 1)], dtype=np.float32)
                    for index, label in enumerate(labels)
                }
            }
        }

    def _store_score_payload(self, mode, bound, labels, bucket_name, suffix=''):
        label_column = self._label_column_for_mode(mode)
        rows = {
            label_column: {str(index): f'{label}{suffix}' for index, label in enumerate(labels)},
            'perturbation score': {str(index): float(len(labels) - index) for index in range(len(labels))},
            'pval_adjusted': {str(index): round(0.01 * (index + 1), 4) for index in range(len(labels))},
        }
        self.adata.uns[f'{mode}_perturbation_score'] = {
            str(bound): {
                bucket_name: {
                    'top_compounds': rows,
                }
            }
        }

    def run(self, mode, bound, inplace=True, **kwargs):
        type(self).run_modes.append((mode, float(bound)))
        type(self).run_call_details.append((mode, float(bound), dict(kwargs)))
        if mode == 'random_pathway_background':
            self._store_delta_payload('random_background_perturbation_deltaD', bound, ['PW1'])
        elif mode == 'pathway':
            self._store_delta_payload('pathway_perturbation_deltaD', bound, self._labels_for_mode('pathway'))
        elif mode == 'random_drug_background':
            self._store_delta_payload(
                'random_background_perturbation_deltaD',
                bound,
                self._normalize_background_labels(kwargs.get('random_genes')),
            )
        elif mode == 'drug':
            self._store_delta_payload('drug_perturbation_deltaD', bound, self._labels_for_mode('drug'))
        elif mode == 'random_background':
            self._store_delta_payload(
                'random_background_perturbation_deltaD',
                bound,
                self._normalize_background_labels(kwargs.get('random_genes')),
            )
        elif mode == 'single_gene':
            self._store_delta_payload('single_gene_perturbation_deltaD', bound, self._labels_for_mode('single_gene'))
        elif mode == 'two_genes':
            self._store_delta_payload('two_genes_perturbation_deltaD', bound, self._labels_for_mode('two_genes'))
        elif mode == 'two_drugs':
            self._store_delta_payload('two_drugs_perturbation_deltaD', bound, self._labels_for_mode('two_drugs'))

    def analysis(self, mode, bound, perturbed_tracks, overall_perturbation_analysis=True, stage=None, direction_mode='two_sided'):
        type(self).analysis_modes.append((mode, float(bound), str(perturbed_tracks), stage))
        is_overall_all = overall_perturbation_analysis and perturbed_tracks == 'all'
        is_grouped_track = overall_perturbation_analysis and perturbed_tracks != 'all'
        bucket_name = 'overall' if is_overall_all or is_grouped_track else '0-1'
        suffix = '' if is_overall_all else ('_group' if is_grouped_track else '_track')
        self._store_score_payload(mode, bound, self._labels_for_mode(mode), bucket_name, suffix=suffix)


def _create_analysis_workspace(tmp_path):
    root = tmp_path / 'workspace'
    stagedata_dir = root / '1' / 'stagedata'
    stagedata_dir.mkdir(parents=True, exist_ok=True)

    adata = AnnData(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]], dtype=np.float32))
    adata.obs = pd.DataFrame(
        {
            'stage': ['0', '1'],
            'leiden': ['0', '1'],
            'ident': ['TYPE_A', 'TYPE_B'],
        },
        index=['cell0', 'cell1'],
    )
    adata.var = pd.DataFrame(index=['GENE1', 'GENE2', 'GENE3'])
    data_path = stagedata_dir / 'dataset.h5ad'
    adata.write(data_path)

    with (stagedata_dir / 'attribute.pkl').open('wb') as handle:
        pickle.dump({'edges': {'0': [(0, 1)]}}, handle)

    (root / '1' / 'idremResults').mkdir(parents=True, exist_ok=True)
    model_dir = root / 'model_save'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'task_1.pth').write_text('fake-model', encoding='utf-8')
    (model_dir / 'training_parameters.json').write_text(
        json.dumps({'task': 'task'}),
        encoding='utf-8',
    )

    cmap_dir = root / 'fake_cmap.npy'
    np.save(cmap_dir, np.array([[1]]))

    target_dir = tmp_path / 'analysis_results'
    return data_path, target_dir, cmap_dir


def _patch_analysis_dependencies(monkeypatch):
    monkeypatch.setattr(analyst_module, 'time', analyst_module.time)
    monkeypatch.setattr(analyst_module.time, 'sleep', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyst_module,
        'calculateDataPathwayOverlapGene',
        lambda adata, customized_pathway=None: _set_pathway_targets(adata),
    )
    monkeypatch.setattr(
        analyst_module,
        'calculateTopPathwayGeneRanking',
        lambda adata: _set_pathway_ranking(adata),
    )
    monkeypatch.setattr(
        analyst_module,
        'find_overlap_and_assign_direction',
        lambda adata, customized_direction=None, customized_drug=None, cmap_dir=None: _set_drug_targets(adata),
    )
    monkeypatch.setattr(analyst_module, 'perturbation', _FakePerturbationRunner)
    monkeypatch.setattr(analyst_module, 'perturbation_centroid', _FakePerturbationRunner)


def _set_pathway_targets(adata):
    adata.uns['data_pathway_overlap_genes'] = {'PW1': ['GENE1']}
    adata.uns['pathway_gene_len_dict'] = {'PW1': 1}
    return adata


def _set_pathway_ranking(adata):
    adata.uns['pathway_ranking'] = {'0': {'0': {'PW1': 1}}}
    return adata


def _set_drug_targets(adata):
    adata.uns['data_drug_overlap_genes'] = {
        'DR1': ['GENE1:+'],
        'DR2': ['GENE2:+'],
        'DR3': ['GENE3:-'],
    }
    adata.uns['drug-gene_len_dict'] = {'DR1': 1, 'DR2': 1, 'DR3': 1}
    return adata


def _load_saved_uns(target_dir: Path):
    with (target_dir / 'attribute.pkl').open('rb') as handle:
        return pickle.load(handle)


def test_single_gpu_optimized_reuses_cached_pathway_and_drug_stages(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturbation_schedule='single_gpu_optimized',
    )

    assert _FakePerturbationRunner.init_count == 1
    assert [mode for mode, _bound in _FakePerturbationRunner.run_modes] == [
        'random_pathway_background',
        'pathway',
        'random_drug_background',
        'drug',
    ]

    _FakePerturbationRunner.run_modes = []
    _FakePerturbationRunner.analysis_modes = []

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturbation_schedule='single_gpu_optimized',
    )

    assert _FakePerturbationRunner.init_count == 1
    assert _FakePerturbationRunner.run_modes == []
    saved_uns = _load_saved_uns(target_dir)
    assert 'pathway_random_background_perturbation_deltaD' in saved_uns
    assert 'drug_random_background_perturbation_deltaD' in saved_uns
    assert saved_uns['analysis_metadata']['perturbation_runner_builds'] == 0


def test_single_gpu_optimized_invalidates_cache_when_perturb_change_changes(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturb_change=0.5,
        perturbation_schedule='single_gpu_optimized',
    )
    first_run_count = len(_FakePerturbationRunner.run_modes)

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturb_change=0.25,
        perturbation_schedule='single_gpu_optimized',
    )

    assert len(_FakePerturbationRunner.run_modes) == first_run_count * 2


def test_single_gpu_optimized_passes_stage_filter_to_pathway_and_drug_scoring(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturb_stage="1",
        perturbation_schedule='single_gpu_optimized',
    )

    assert ("pathway", 0.5, "all", 1) in _FakePerturbationRunner.analysis_modes
    assert ("drug", 0.5, "all", 1) in _FakePerturbationRunner.analysis_modes
    saved_uns = _load_saved_uns(target_dir)
    assert saved_uns["analysis_metadata"]["perturb_stage"] == "1"


def test_single_gpu_optimized_stage_background_cache_is_separate_from_overall(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturbation_schedule='single_gpu_optimized',
    )

    _FakePerturbationRunner.run_modes = []
    _FakePerturbationRunner.analysis_modes = []

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturb_stage="1",
        perturbation_schedule='single_gpu_optimized',
    )

    assert [mode for mode, _bound in _FakePerturbationRunner.run_modes] == [
        'random_pathway_background',
        'pathway',
        'random_drug_background',
        'drug',
    ]


def test_single_gpu_optimized_persists_overall_and_grouped_track_scores(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        perturbation_schedule='single_gpu_optimized',
    )

    assert ("pathway", 0.5, "all", None) in _FakePerturbationRunner.analysis_modes
    assert ("drug", 0.5, "all", None) in _FakePerturbationRunner.analysis_modes
    saved_uns = _load_saved_uns(target_dir)
    pathway_bucket = saved_uns["pathway_perturbation_score"]["0.5"]
    drug_bucket = saved_uns["drug_perturbation_score"]["0.5"]
    assert {"overall", "0"} <= set(pathway_bucket.keys())
    assert {"overall", "0"} <= set(drug_bucket.keys())
    assert pathway_bucket["0"]["top_compounds"]["compound"]["0"] == "PW1_group"
    assert drug_bucket["0"]["top_compounds"]["compound"]["0"] == "DR1_group"


def test_core_perturbation_results_store_raw_and_fdr_adjusted_pvalues():
    adata = AnnData(np.zeros((1, 1), dtype=np.float32))
    adata.obs = pd.DataFrame(index=["cell0"])
    adata.var = pd.DataFrame(index=["GENE1"])
    analysis = perturbationAnalysis(adata, ".", log2fc=0.5, mode="pathway")

    analysis.calculateAvg = lambda objectdic, sanity=False, tracklist='all': objectdic
    analysis.conver_object_track_score_to_track_object_score = (
        lambda pathways, perturbationresultdic, sanity_perturbationresultdic, scoreindex, track: (
            {'overall': [1.0, 0.8, 0.7]},
            {'overall': [0.1, 0.2, 0.3]},
        )
    )
    analysis.getTendencyFromIDREM = lambda: {'1': np.array([1.0])}
    score_lookup = {
        "PathA": (1.0, 0.01),
        "PathB": (0.8, 0.04),
        "PathC": (0.7, 0.03),
    }
    analysis.getTrackObjectCDF = lambda eachpathway, *args, **kwargs: score_lookup[eachpathway]

    objectranking = {("overall", "top_compounds"): ["PathA", "PathB", "PathC"]}
    objectdic = {"PathA": {}, "PathB": {}, "PathC": {}}
    gene_in_object = {
        "PathA": ["GENE1:+"],
        "PathB": ["GENE1:+"],
        "PathC": ["GENE1:+"],
    }

    results = analysis.getSummarizedResults(
        "all",
        objectranking,
        objectdic,
        objectdic,
        {"0-1": 1.0},
        gene_in_object,
        "avg_backScore",
        {"GENE1": 1.0},
        True,
    )

    top = results["overall"]["top_compounds"]
    raw = [top["pval"][str(index)] for index in range(3)]
    adjusted = [top["pval_adjusted"][str(index)] for index in range(3)]

    assert raw == pytest.approx([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx(_bh_adjust(raw))


def test_resource_aware_schedule_is_reserved(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    with pytest.raises(NotImplementedError):
        analyst.start_analyse(
            progressionmarker_background_sampling=3,
            run_perturbation=True,
            random_times=3,
            ignore_dynamic_markers=True,
            ignore_hcmarkers=True,
            perturbation_schedule='resource_aware',
        )


def test_cpu_exact_pathway_stage_uses_parallel_executor_when_gpu_is_unavailable(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()
    captured = {}

    def _fake_parallel(**kwargs):
        captured.update(kwargs)
        return {"0.5": {"0-1": {"PW1": np.array([2.0])}}}

    monkeypatch.setattr(analyst_module, "run_exact_parallel_delta", _fake_parallel)
    monkeypatch.setattr(analyst_module.analyst, "_gpu_available", staticmethod(lambda: False))

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        centroid=False,
        cpu_workers=4,
    )

    assert captured["backend_mode"] == "drug"
    assert captured["cpu_workers"] == 4
    saved_uns = _load_saved_uns(target_dir)
    assert saved_uns["analysis_metadata"]["resolved_execution_backend"] == "cpu_exact_parallel"
    assert saved_uns["analysis_metadata"]["cpu_workers"] == 4


def test_cpu_centroid_stage_stays_serial_when_centroid_is_selected(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    called = {"parallel": 0}

    def _fake_parallel(**kwargs):
        called["parallel"] += 1
        return {}

    monkeypatch.setattr(analyst_module, "run_exact_parallel_delta", _fake_parallel)
    monkeypatch.setattr(analyst_module.analyst, "_gpu_available", staticmethod(lambda: False))

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        centroid=True,
    )

    assert called["parallel"] == 0
    saved_uns = _load_saved_uns(target_dir)
    assert saved_uns["analysis_metadata"]["resolved_execution_backend"] == "cpu_centroid"


def test_single_and_combination_perturbations_persist_mode_specific_backgrounds_and_grouped_scores(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        ignore_pathway_perturabtion=True,
        ignore_drug_perturabtion=True,
        run_single_gene_perturbation=True,
        run_two_gene_perturbation=True,
        run_two_drug_perturbation=True,
        perturbation_schedule='single_gpu_optimized',
    )

    assert [mode for mode, _bound in _FakePerturbationRunner.run_modes] == [
        'random_background',
        'single_gene',
        'random_background',
        'two_genes',
        'random_drug_background',
        'two_drugs',
    ]

    saved_uns = _load_saved_uns(target_dir)
    assert 'single_gene_random_background_perturbation_deltaD' in saved_uns
    assert 'two_genes_random_background_perturbation_deltaD' in saved_uns
    assert 'two_drugs_random_background_perturbation_deltaD' in saved_uns
    assert {'overall', '0'} <= set(saved_uns['single_gene_perturbation_score']['0.5'].keys())
    assert {'overall', '0'} <= set(saved_uns['two_genes_perturbation_score']['0.5'].keys())
    assert {'overall', '0'} <= set(saved_uns['two_drugs_perturbation_score']['0.5'].keys())
    assert len(saved_uns['combinatorial_perturbation_genes_set1']) == 3
    assert len(saved_uns['two_genes_perturbation_deltaD']['0.5']['0-1']) == 6
    assert len(saved_uns['data_two_drug_overlap_genes']) == 3


def test_drug_background_uses_minimum_1000_samples_without_changing_combo_background_counts(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        run_single_gene_perturbation=True,
        run_two_gene_perturbation=True,
        run_two_drug_perturbation=True,
        perturbation_schedule='single_gpu_optimized',
    )

    random_call_times = {}
    for mode, _bound, kwargs in _FakePerturbationRunner.run_call_details:
        if mode in {'random_drug_background', 'random_background'}:
            random_call_times.setdefault(mode, []).append(int(kwargs.get('random_times', -1)))

    assert random_call_times['random_drug_background'] == [1000, 3]
    assert random_call_times['random_background'] == [3, 3]

    saved_uns = _load_saved_uns(target_dir)
    assert saved_uns['analysis_metadata']['random_background_sampling_times'] == 3
    assert saved_uns['analysis_metadata']['effective_drug_background_sampling_times'] == 1000


def test_shortlisted_combination_modes_reuse_single_mode_prerequisites(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=True,
        ignore_hcmarkers=True,
        ignore_pathway_perturabtion=True,
        ignore_drug_perturabtion=True,
        run_two_gene_perturbation=True,
        run_two_drug_perturbation=True,
        shortlist_two_gene_candidates=True,
        shortlist_two_drug_candidates=True,
        top_n_single_gene_candidates=2,
        top_n_two_drug_candidates=2,
        perturbation_schedule='single_gpu_optimized',
    )

    assert [mode for mode, _bound in _FakePerturbationRunner.run_modes] == [
        'random_drug_background',
        'drug',
        'random_background',
        'single_gene',
        'random_background',
        'two_genes',
        'random_drug_background',
        'two_drugs',
    ]

    saved_uns = _load_saved_uns(target_dir)
    assert saved_uns['combinatorial_perturbation_genes_set1'] == ['GENE1', 'GENE2']
    assert len(saved_uns['two_genes_perturbation_deltaD']['0.5']['0-1']) == 4
    assert list(saved_uns['data_two_drug_overlap_genes'].keys()) == ['DR1__DR2']
    assert saved_uns['analysis_metadata']['shortlist_two_gene_candidates'] is True
    assert saved_uns['analysis_metadata']['shortlist_two_drug_candidates'] is True
