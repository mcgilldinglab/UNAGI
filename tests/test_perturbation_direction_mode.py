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

    def perturb_input_data(self, cells, gene_input_dict, mode="direct"):
        self.calls.append(dict(gene_input_dict))
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
    elif mode == "two_genes":
        runner.adata.uns["combinatorial_perturbation_genes_set1"] = ["GENE1"]
    runner.tracks = {1: None}
    runner.stageadata = [None, None]
    runner.pb = _DummyPB()
    runner.progress_logger = _DummyProgress()
    runner.getTrack = lambda stage, cluster: [[0], [1]]
    return runner


@pytest.mark.parametrize(
    ("mode", "expected_one_sided_calls", "expected_first_call"),
    [
        ("pathway", 1, {"GENE1": 0.5, "GENE2": 0.5}),
        ("drug", 1, {"GENE1": 0.5, "GENE2": 2.0}),
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
