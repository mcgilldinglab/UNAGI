from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import UNAGI.perturbations.perturbation as perturbation_module


class _DummyPB:
    def __init__(self):
        self.calls = []
        self.data = SimpleNamespace(obsm={})

    def compile_direct_targets(self, gene_input_dict):
        return tuple(gene_input_dict.items())

    def perturb_input_data(self, cells, gene_input_dict, mode="direct"):
        self.calls.append(dict(gene_input_dict))
        self.data.obsm["perturbed_embedding"] = np.zeros((2, 2), dtype=float)

    def perturb_input_data_compiled(self, cells, compiled_targets, mode="direct"):
        self.calls.append(dict(compiled_targets))
        self.data.obsm["perturbed_embedding"] = np.zeros((2, 2), dtype=float)

    def calculate_deltaD_for_tracks(self, track_indices):
        return [np.array([0.0, 1.0]), np.array([1.0, 0.0])]


class _DummyProgress:
    def log(self, *args, **kwargs):
        return None

    def should_report(self, finished, total):
        return False

    def progress(self, *args, **kwargs):
        return None


def _delta_from_compiled(compiled_targets, track_indices):
    weighted_sum = 0.0
    for item_index, (gene_name, factor) in enumerate(compiled_targets):
        weighted_sum += (item_index + 1) * (sum(ord(ch) for ch in gene_name) + float(factor))
    size = len(track_indices)
    delta = np.zeros((size, size), dtype=float)
    for row_index in range(size):
        for col_index in range(size):
            if row_index == col_index:
                continue
            delta[row_index, col_index] = weighted_sum + (row_index + 1) * 0.1 + (col_index + 1) * 0.01
    return delta


class _DummyCompiledPB:
    def __init__(self):
        self.serial_calls = []
        self.batched_calls = []
        self.track_cache_calls = []
        self.config = {"GCN": False}
        self.data = SimpleNamespace(obsm={})

    def compile_direct_targets(self, gene_input_dict):
        return tuple((gene_name, float(gene_input_dict[gene_name])) for gene_name in sorted(gene_input_dict))

    def prepare_track_cache(self, rows):
        self.track_cache_calls.append(tuple(tuple(track_rows) for track_rows in rows))
        return {"rows": rows}

    def perturb_input_data_compiled(self, cells, compiled_targets, mode="direct"):
        self.serial_calls.append(compiled_targets)
        self.data.obsm["perturbed_embedding"] = compiled_targets

    def calculate_deltaD_for_tracks(self, track_indices):
        return _delta_from_compiled(self.data.obsm["perturbed_embedding"], track_indices)

    def perturb_input_data_compiled_batched(self, compiled_targets_batch):
        self.batched_calls.append(tuple(compiled_targets_batch))
        return list(compiled_targets_batch), "hit"

    def calculate_deltaD_for_tracks_batched(self, perturbed_embedding_batch, prepared_track_cache):
        return np.stack(
            [
                _delta_from_compiled(compiled_targets, prepared_track_cache["rows"])
                for compiled_targets in perturbed_embedding_batch
            ],
            axis=0,
        )


def _make_runner(mode: str):
    runner = object.__new__(perturbation_module.perturbation)
    runner.adata = SimpleNamespace(
        obs=pd.DataFrame(
            {"stage": ["0", "1"], "leiden": ["0", "1"]},
            index=["cell0", "cell1"],
        ),
        uns={},
        var_names=pd.Index(["GENE1", "GENE2"]),
    )
    if mode == "pathway":
        runner.adata.uns["data_pathway_overlap_genes"] = {"PWY1": ["GENE1", "GENE2"]}
    elif mode == "drug":
        runner.adata.uns["data_drug_overlap_genes"] = {"DRUG1": ["GENE1:+", "GENE2:-"]}
    elif mode == "two_drugs":
        runner.adata.uns["data_two_drug_overlap_genes"] = {"DRUG1__DRUG2": ["GENE1:+", "GENE2:-"]}
    elif mode == "two_genes":
        runner.adata.uns["combinatorial_perturbation_genes_set1"] = ["GENE1"]
    runner.tracks = {1: None}
    runner.stageadata = [None, None]
    runner.pb = _DummyPB()
    runner.progress_logger = _DummyProgress()
    runner.getTrack = lambda stage, cluster: [[0], [1]]
    runner.gpu_exact_force_legacy = False
    runner.gpu_exact_batch_size = None
    runner.last_gpu_batch_meta = None
    runner._legacy_track_cache = {}
    return runner


def _make_two_gene_exact_runner(*, force_legacy: bool):
    runner = object.__new__(perturbation_module.perturbation)
    runner.adata = SimpleNamespace(
        obs=pd.DataFrame(
            {"stage": ["0", "1"], "leiden": ["0", "1"]},
            index=["cell0", "cell1"],
        ),
        uns={"combinatorial_perturbation_genes_set1": ["GENE1"]},
        var_names=pd.Index(["GENE1", "GENE2", "GENE3"]),
    )
    runner.tracks = {1: None}
    runner.stageadata = [None, None]
    runner.pb = _DummyCompiledPB()
    runner.progress_logger = _DummyProgress()
    runner.getTrack = lambda stage, cluster: [[0], [1]]
    runner.gpu_exact_force_legacy = force_legacy
    runner.gpu_exact_batch_size = None
    runner.last_gpu_batch_meta = None
    runner._legacy_track_cache = {}
    return runner


def _make_two_drug_exact_runner(*, force_legacy: bool):
    runner = object.__new__(perturbation_module.perturbation)
    runner.adata = SimpleNamespace(
        obs=pd.DataFrame(
            {"stage": ["0", "1"], "leiden": ["0", "1"]},
            index=["cell0", "cell1"],
        ),
        uns={
            "data_two_drug_overlap_genes": {
                "DRUG1__DRUG2": ["GENE1:+", "GENE2:-"],
                "DRUG1__DRUG3": ["GENE1:+", "GENE3:+"],
            }
        },
        var_names=pd.Index(["GENE1", "GENE2", "GENE3"]),
    )
    runner.tracks = {1: None}
    runner.stageadata = [None, None]
    runner.pb = _DummyCompiledPB()
    runner.progress_logger = _DummyProgress()
    runner.getTrack = lambda stage, cluster: [[0], [1]]
    runner.gpu_exact_force_legacy = force_legacy
    runner.gpu_exact_batch_size = None
    runner.last_gpu_batch_meta = None
    runner._legacy_track_cache = {}
    return runner


def _assert_nested_delta_payload_equal(left_payload, right_payload):
    assert set(left_payload.keys()) == set(right_payload.keys())
    for direction_key in left_payload:
        assert set(left_payload[direction_key].keys()) == set(right_payload[direction_key].keys())
        for track_name in left_payload[direction_key]:
            assert set(left_payload[direction_key][track_name].keys()) == set(
                right_payload[direction_key][track_name].keys()
            )
            for item_name in left_payload[direction_key][track_name]:
                np.testing.assert_allclose(
                    left_payload[direction_key][track_name][item_name],
                    right_payload[direction_key][track_name][item_name],
                )


@pytest.mark.parametrize(
    ("mode", "expected_one_sided_calls", "expected_first_call"),
    [
        ("pathway", 1, {"GENE1": 0.5, "GENE2": 0.5}),
        ("drug", 1, {"GENE1": 0.5, "GENE2": 2.0}),
        ("two_drugs", 1, {"GENE1": 0.5, "GENE2": 2.0}),
        ("single_gene", 2, {"GENE1": 0.5}),
        ("two_genes", 1, {"GENE1": 0.5, "GENE2": 0.5}),
    ],
)
def test_run_direction_mode_controls_forward_and_reverse(mode, expected_one_sided_calls, expected_first_call):
    one_sided = _make_runner(mode)
    one_sided.run(mode, 0.5, inplace=True, direction_mode="one_sided")

    assert len(one_sided.pb.calls) == expected_one_sided_calls
    assert one_sided.pb.calls[0] == expected_first_call
    assert set(one_sided.adata.uns[f"{mode}_perturbation_deltaD"].keys()) == {"0.5"}

    two_sided = _make_runner(mode)
    two_sided.run(mode, 0.5, inplace=True)

    assert len(two_sided.pb.calls) == expected_one_sided_calls * 2
    assert set(two_sided.adata.uns[f"{mode}_perturbation_deltaD"].keys()) == {"0.5", "2.0"}


@pytest.mark.parametrize(
    ("direction_mode", "expected_keys"),
    [
        ("one_sided", {"0.5"}),
        (None, {"0.5", "2.0"}),
    ],
)
def test_analysis_stores_requested_direction_keys(monkeypatch, direction_mode, expected_keys):
    class FakeAnalysis:
        def __init__(self, adata, target_directory, log2fc, stage=None, mode=None, direction_mode=None):
            self.direction_mode = direction_mode or "two_sided"

        def main_analysis(self, perturbed_tracks, overall_perturbation_analysis, score=None, items=None):
            return {"overall": {"top_compounds": {"0": {"compound": "X", "pval_adjusted": 0.01}}}}

    monkeypatch.setattr(perturbation_module, "perturbationAnalysis", FakeAnalysis)

    runner = object.__new__(perturbation_module.perturbation)
    runner.adata = SimpleNamespace(
        obs=pd.DataFrame({"stage": ["0"], "leiden": ["0"]}),
        uns={},
    )
    runner.idrem_dir = Path("/tmp")

    kwargs = {}
    if direction_mode is not None:
        kwargs["direction_mode"] = direction_mode
    runner.analysis("pathway", 0.5, **kwargs)

    assert set(runner.adata.uns["pathway_perturbation_score"].keys()) == expected_keys


@pytest.mark.parametrize(
    ("direction_mode", "expected_keys"),
    [
        ("one_sided", {"0.5"}),
        (None, {"0.5", "2.0"}),
    ],
)
def test_two_genes_gpu_exact_matches_legacy_path(direction_mode, expected_keys):
    optimized = _make_two_gene_exact_runner(force_legacy=False)
    legacy = _make_two_gene_exact_runner(force_legacy=True)

    kwargs = {"inplace": True, "CUDA": True, "device": "cuda:0"}
    if direction_mode is not None:
        kwargs["direction_mode"] = direction_mode

    optimized.run("two_genes", 0.5, **kwargs)
    legacy.run("two_genes", 0.5, **kwargs)

    optimized_payload = optimized.adata.uns["two_genes_perturbation_deltaD"]
    legacy_payload = legacy.adata.uns["two_genes_perturbation_deltaD"]

    assert set(optimized_payload.keys()) == expected_keys
    assert set(legacy_payload.keys()) == expected_keys
    _assert_nested_delta_payload_equal(optimized_payload, legacy_payload)

    assert optimized.pb.batched_calls
    assert not optimized.pb.serial_calls
    assert optimized.pb.track_cache_calls
    assert optimized.last_gpu_batch_meta["batched"] is True
    assert optimized.last_gpu_batch_meta["fallback"] is False

    assert legacy.pb.serial_calls
    assert not legacy.pb.batched_calls
    assert not legacy.pb.track_cache_calls
    assert legacy.last_gpu_batch_meta is None


def test_two_genes_gpu_exact_force_legacy_disables_batching():
    runner = _make_two_gene_exact_runner(force_legacy=True)

    runner.run("two_genes", 0.5, inplace=True, CUDA=True, device="cuda:0")

    assert runner.pb.serial_calls
    assert not runner.pb.batched_calls
    assert runner.last_gpu_batch_meta is None


def test_two_drugs_gpu_exact_matches_legacy_path():
    optimized = _make_two_drug_exact_runner(force_legacy=False)
    legacy = _make_two_drug_exact_runner(force_legacy=True)

    optimized.run("two_drugs", 0.5, inplace=True, CUDA=True, device="cuda:0", direction_mode="one_sided")
    legacy.run("two_drugs", 0.5, inplace=True, CUDA=True, device="cuda:0", direction_mode="one_sided")

    optimized_payload = optimized.adata.uns["two_drugs_perturbation_deltaD"]
    legacy_payload = legacy.adata.uns["two_drugs_perturbation_deltaD"]

    assert set(optimized_payload.keys()) == {"0.5"}
    assert set(legacy_payload.keys()) == {"0.5"}
    _assert_nested_delta_payload_equal(optimized_payload, legacy_payload)
    assert optimized.pb.batched_calls
    assert not optimized.pb.serial_calls
    assert legacy.pb.serial_calls
    assert not legacy.pb.batched_calls


def test_get_unique_random_genes_avoids_duplicates_when_possible():
    adata = SimpleNamespace(var=SimpleNamespace(index=pd.Index(["GENE1", "GENE2", "GENE3"])))
    sample = perturbation_module.get_unique_random_genes(adata, 3, seed=7)
    assert len(set(sample.tolist())) == 3
