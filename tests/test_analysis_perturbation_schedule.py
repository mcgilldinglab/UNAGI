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
    analysis_modes = []

    def __init__(self, adata, model_name, idrem_dir):
        type(self).init_count += 1
        self.adata = adata

    @classmethod
    def reset(cls):
        cls.init_count = 0
        cls.run_modes = []
        cls.analysis_modes = []

    def run(self, mode, bound, inplace=True, **kwargs):
        type(self).run_modes.append((mode, float(bound)))
        key = str(bound)
        if mode == 'random_pathway_background':
            self.adata.uns['random_background_perturbation_deltaD'] = {
                key: {'0-1': {'PW1': np.array([1.0])}}
            }
        elif mode == 'pathway':
            self.adata.uns['pathway_perturbation_deltaD'] = {
                key: {'0-1': {'PW1': np.array([2.0])}}
            }
        elif mode == 'random_drug_background':
            self.adata.uns['random_background_perturbation_deltaD'] = {
                key: {'0-1': {'DR1': np.array([3.0])}}
            }
        elif mode == 'drug':
            self.adata.uns['drug_perturbation_deltaD'] = {
                key: {'0-1': {'DR1': np.array([4.0])}}
            }

    def analysis(self, mode, bound, perturbed_tracks, overall_perturbation_analysis=True, stage=None, direction_mode='two_sided'):
        type(self).analysis_modes.append((mode, float(bound), str(perturbed_tracks), stage))
        key = str(bound)
        is_overall_all = overall_perturbation_analysis and perturbed_tracks == 'all'
        is_grouped_track = overall_perturbation_analysis and perturbed_tracks != 'all'
        if mode == 'pathway':
            self.adata.uns['pathway_perturbation_score'] = {
                key: {
                    (
                        'overall'
                        if is_overall_all or is_grouped_track
                        else '0-1'
                    ): {
                        'top_compounds': {
                            'compound': {'0': 'PW1' if is_overall_all else ('PW1_group' if is_grouped_track else 'PW1_track')},
                            'perturbation score': {'0': 1.0 if is_overall_all else (0.9 if is_grouped_track else 0.8)},
                            'pval_adjusted': {'0': 0.05 if is_overall_all else (0.06 if is_grouped_track else 0.07)},
                        }
                    }
                }
            }
        elif mode == 'drug':
            self.adata.uns['drug_perturbation_score'] = {
                key: {
                    (
                        'overall'
                        if is_overall_all or is_grouped_track
                        else '0-1'
                    ): {
                        'top_compounds': {
                            'compound': {'0': 'DR1' if is_overall_all else ('DR1_group' if is_grouped_track else 'DR1_track')},
                            'perturbation score': {'0': 2.0 if is_overall_all else (1.8 if is_grouped_track else 1.6)},
                            'pval_adjusted': {'0': 0.01 if is_overall_all else (0.02 if is_grouped_track else 0.03)},
                        }
                    }
                }
            }


def _create_analysis_workspace(tmp_path):
    root = tmp_path / 'workspace'
    stagedata_dir = root / '1' / 'stagedata'
    stagedata_dir.mkdir(parents=True, exist_ok=True)

    adata = AnnData(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    adata.obs = pd.DataFrame(
        {
            'stage': ['0', '1'],
            'leiden': ['0', '1'],
            'ident': ['TYPE_A', 'TYPE_B'],
        },
        index=['cell0', 'cell1'],
    )
    adata.var = pd.DataFrame(index=['GENE1', 'GENE2'])
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
    adata.uns['data_drug_overlap_genes'] = {'DR1': ['GENE1:+']}
    adata.uns['drug-gene_len_dict'] = {'DR1': 1}
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
