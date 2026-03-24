import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import UNAGI.UNAGI_analyst as analyst_module
import UNAGI.marker_discovery.dynamic_markers as dynamic_markers_module
from UNAGI.marker_discovery.dynamic_markers import runGetProgressionMarker_one_dist
from UNAGI.marker_discovery.dynamic_markers_helper import (
    _compare_nested,
    compare_progressionmarker_strategies,
    get_progressionmarker_background,
)


def _make_dynamic_adata() -> AnnData:
    x = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.5, 0.2],
                [0.9, 0.1, 0.4, 0.3],
                [0.2, 1.1, 0.3, 0.6],
                [0.1, 1.0, 0.4, 0.7],
                [0.6, 0.4, 1.2, 0.1],
                [0.5, 0.3, 1.1, 0.2],
            ],
            dtype=np.float32,
        )
    )
    adata = AnnData(x)
    adata.obs = pd.DataFrame(
        {
            'stage': ['0', '0', '1', '1', '2', '2'],
            'leiden': ['0', '1', '0', '1', '0', '1'],
            'ident': ['A', 'B', 'A', 'B', 'A', 'B'],
        },
        index=[f'cell{i}' for i in range(6)],
    )
    adata.var = pd.DataFrame(index=['GENE0', 'GENE1', 'GENE2', 'GENE3'])
    adata.uns['edges'] = {
        '0': [(0, 0), (1, 1)],
        '1': [(0, 0), (1, 1)],
    }
    return adata


def _fake_idrem_table(filename: str):
    if filename.startswith('0-0-0'):
        stage_rows = [
            ['GENE0', 0.4, 0.8, 1.2],
            ['GENE1', 1.2, 0.8, 0.4],
            ['GENE2', 0.1, 0.2, 0.4],
            ['GENE3', 0.7, 0.3, 0.1],
        ]
    else:
        stage_rows = [
            ['GENE0', 0.2, 0.4, 0.7],
            ['GENE1', 0.9, 0.6, 0.3],
            ['GENE2', 0.3, 0.7, 1.0],
            ['GENE3', 0.8, 0.5, 0.2],
        ]
    tt = [None] * 9
    tt[8] = np.array(
        [['gene', 'stage0', 'stage1', 'stage2'], *stage_rows],
        dtype=object,
    )
    return tt


def _prepare_idrem_dir(tmp_path: Path) -> Path:
    idrem_dir = tmp_path / 'idremResults'
    (idrem_dir / '0-0-0.txt_viz').mkdir(parents=True, exist_ok=True)
    (idrem_dir / '1-1-1.txt_viz').mkdir(parents=True, exist_ok=True)
    return idrem_dir


def test_progressionmarker_background_optimized_matches_legacy_serial_and_parallel(tmp_path):
    adata = _make_dynamic_adata()
    cache_dir = tmp_path / 'cache'

    np.random.seed(123)
    legacy = get_progressionmarker_background(4, adata, 3, strategy='legacy')
    np.random.seed(123)
    optimized_serial = get_progressionmarker_background(
        4,
        adata,
        3,
        strategy='optimized',
        cache_dir=cache_dir,
        force_serial=True,
        cpu_workers=1,
    )
    np.random.seed(123)
    optimized_parallel = get_progressionmarker_background(
        4,
        adata,
        3,
        strategy='optimized',
        cache_dir=cache_dir,
        force_serial=False,
        cpu_workers=2,
    )

    assert _compare_nested(legacy, optimized_serial, atol=1e-8, rtol=1e-6)[0]
    assert _compare_nested(legacy, optimized_parallel, atol=1e-8, rtol=1e-6)[0]
    assert (cache_dir / 'progressionmarker_stage_cache_manifest.json').exists()


def test_runGetProgressionMarker_one_dist_optimized_matches_legacy(tmp_path, monkeypatch):
    idrem_dir = _prepare_idrem_dir(tmp_path)
    monkeypatch.setattr(dynamic_markers_module, 'readIdremJson', lambda path, filename: _fake_idrem_table(filename))
    background = {
        '0-0-0': [np.array([0.4, -0.2, 0.3, -0.1]), np.array([0.3, -0.3, 0.2, -0.2])],
        '1-1-1': [np.array([0.5, -0.1, 0.4, -0.2]), np.array([0.2, -0.4, 0.1, -0.3])],
    }

    legacy = runGetProgressionMarker_one_dist(str(idrem_dir), background, 4, cutoff=0.05, strategy='legacy')
    optimized = runGetProgressionMarker_one_dist(str(idrem_dir), background, 4, cutoff=0.05, strategy='optimized')

    assert _compare_nested(legacy, optimized, atol=1e-8, rtol=1e-6)[0]


def test_compare_progressionmarker_strategies_reports_ok(tmp_path, monkeypatch):
    adata = _make_dynamic_adata()
    idrem_dir = _prepare_idrem_dir(tmp_path)
    monkeypatch.setattr(dynamic_markers_module, 'readIdremJson', lambda path, filename: _fake_idrem_table(filename))

    comparison = compare_progressionmarker_strategies(
        times=3,
        adata=adata,
        total_stage=3,
        idrem_directory=idrem_dir,
        size=adata.shape[1],
        cache_dir=tmp_path / 'cache',
        cpu_workers=2,
    )

    assert comparison['ok'] is True
    assert all(section['ok'] for section in comparison['comparisons'].values())


def _prepare_analysis_workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / 'workspace'
    stagedata_dir = root / '1' / 'stagedata'
    stagedata_dir.mkdir(parents=True, exist_ok=True)
    adata = _make_dynamic_adata()
    data_path = stagedata_dir / 'dataset.h5ad'
    adata.write(data_path)
    with (stagedata_dir / 'attribute.pkl').open('wb') as handle:
        pickle.dump({'edges': {'0': [(0, 0), (1, 1)], '1': [(0, 0), (1, 1)]}}, handle)
    (root / '1' / 'idremResults').mkdir(parents=True, exist_ok=True)
    model_dir = root / 'model_save'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'example_task_1.pth').write_text('fake-model', encoding='utf-8')
    (model_dir / 'training_parameters.json').write_text(json.dumps({'task': 'example_task'}), encoding='utf-8')
    target_dir = tmp_path / 'analysis_results'
    return data_path, target_dir


def test_start_analyse_passes_dynamic_marker_strategy_controls(monkeypatch, tmp_path):
    data_path, target_dir = _prepare_analysis_workspace(tmp_path)
    captured = {}

    monkeypatch.setattr(analyst_module.time, 'sleep', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analyst_module, 'calculateDataPathwayOverlapGene', lambda adata, customized_pathway=None: adata)
    monkeypatch.setattr(analyst_module, 'calculateTopPathwayGeneRanking', lambda adata: adata)
    monkeypatch.setattr(
        analyst_module,
        'get_progressionmarker_background',
        lambda *args, **kwargs: (
            captured.__setitem__('background_kwargs', kwargs) or {'0-0-0': [np.zeros(4)]}
        ),
    )
    monkeypatch.setattr(
        analyst_module,
        'runGetProgressionMarker_one_dist',
        lambda *args, **kwargs: (
            captured.__setitem__('marker_kwargs', kwargs) or {'0-0-0': {'increasing': {}, 'decreasing': {}}}
        ),
    )

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=False,
        random_times=3,
        ignore_hcmarkers=True,
        dynamic_marker_strategy='optimized',
        dynamic_marker_cpu_workers=3,
        dynamic_marker_force_serial=True,
    )

    assert captured['background_kwargs']['strategy'] == 'optimized'
    assert captured['background_kwargs']['cpu_workers'] == 3
    assert captured['background_kwargs']['force_serial'] is True
    assert captured['marker_kwargs']['strategy'] == 'optimized'
