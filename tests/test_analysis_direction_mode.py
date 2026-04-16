from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from UNAGI.perturbations.analysis_perturbation import perturbationAnalysis as StandardAnalysis
from UNAGI.perturbations.normalize_analysis_perturbation import perturbationAnalysis as NormalizedAnalysis


def _make_adata(include_reverse: bool):
    forward = {
        "0-1": {
            "ITEM1": [
                np.array([0.0, 1.0]),
                np.array([2.0, 0.0]),
            ]
        }
    }
    reverse = {
        "0-1": {
            "ITEM1": [
                np.array([0.0, 0.5]),
                np.array([0.5, 0.0]),
            ]
        }
    }
    uns = {
        "pathway_perturbation_deltaD": {"0.5": forward},
        "random_background_perturbation_deltaD": {"0.5": forward},
    }
    if include_reverse:
        uns["pathway_perturbation_deltaD"]["2.0"] = reverse
        uns["random_background_perturbation_deltaD"]["2.0"] = reverse
    return SimpleNamespace(uns=uns, var=pd.DataFrame(index=["GENE1"]))


def _patch_score(monkeypatch, analysis):
    if isinstance(analysis, StandardAnalysis):
        monkeypatch.setattr(
            analysis,
            "calculateScore",
            lambda delta, flag, weight=1: (float(np.sum(delta)), 0.0),
        )
    else:
        monkeypatch.setattr(
            analysis,
            "calculateScore",
            lambda delta, flag, mean_distance_change=None, std_distance_change=None, weight=1: (float(np.sum(delta)), 0.0),
        )


def _make_track_weight_adata():
    counts_by_stage = {
        "0": {"0": 6},
        "1": {"0": 6, "14": 2},
        "2": {"1": 7, "15": 1},
        "3": {"0": 8, "10": 2, "18": 1},
    }
    rows = []
    index = []
    for stage, cluster_counts in counts_by_stage.items():
        for cluster, count in cluster_counts.items():
            for replicate in range(count):
                index.append(f"cell_{stage}_{cluster}_{replicate}")
                rows.append({"stage": str(stage), "leiden": str(cluster)})
    obs = pd.DataFrame(rows, index=index)
    return AnnData(
        np.zeros((len(obs), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["GENE1"]),
    )


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_one_sided_load_uses_single_direction_only(monkeypatch, analysis_cls):
    analysis = analysis_cls(
        _make_adata(include_reverse=False),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
        direction_mode="one_sided",
    )
    _patch_score(monkeypatch, analysis)

    result = analysis.load(
        {"ITEM1": ["GENE1"]},
        {"0-1": 1.0},
        score_weight=1,
        overall_perturbation_analysis=True,
        sanity=False,
    )

    assert analysis.direction_mode == "one_sided"
    assert result["ITEM1"]["overall"] == pytest.approx(1.5)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_two_sided_load_combines_forward_and_reverse(monkeypatch, analysis_cls):
    analysis = analysis_cls(
        _make_adata(include_reverse=True),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
    )
    _patch_score(monkeypatch, analysis)

    result = analysis.load(
        {"ITEM1": ["GENE1"]},
        {"0-1": 1.0},
        score_weight=1,
        overall_perturbation_analysis=True,
        sanity=False,
    )

    assert analysis.direction_mode == "two_sided"
    assert result["ITEM1"]["overall"] == pytest.approx(0.5)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_one_sided_filter_keeps_negative_scores(analysis_cls):
    analysis = analysis_cls(
        _make_adata(include_reverse=False),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
        direction_mode="one_sided",
    )

    names = ["ITEM_POS", "ITEM_NEG"]
    scores = np.array([0.2, -0.3])
    returned = analysis.fitlerOutNarrowPathway(scores, np.array([0.0, 0.1]), names, names)

    assert list(returned) == names


def test_normalized_main_analysis_uses_sanity_distance_stats(monkeypatch):
    adata = SimpleNamespace(
        uns={"data_drug_overlap_genes": {"DrugA": ["GENE1:+"]}},
        var=pd.DataFrame(index=["GENE1"]),
    )
    analysis = NormalizedAnalysis(
        adata,
        target_directory=".",
        log2fc=0.5,
        mode="drug",
    )

    monkeypatch.setattr(analysis, "get_tracks", lambda: ["0-1"])
    monkeypatch.setattr(analysis, "get_track_weight", lambda tracks, perturbed_tracks: {"0-1": 1.0})
    monkeypatch.setattr(analysis, "getTendencyDict", lambda track_weight: {"GENE1": 1.0})
    monkeypatch.setattr(analysis, "getTopDownObjects", lambda *args, **kwargs: {"overall": {}})
    monkeypatch.setattr(analysis, "getSummarizedResults", lambda *args, **kwargs: {"ok": True})

    load_calls = []

    def fake_load(
        data_pathway_overlap_genes,
        track_weight,
        score_weight,
        overall_perturbation_analysis=True,
        sanity=False,
        mean_distance_change=None,
        std_distance_change=None,
    ):
        load_calls.append(
            {
                "sanity": sanity,
                "weight": score_weight,
                "mean_distance_change": mean_distance_change,
                "std_distance_change": std_distance_change,
                "items": data_pathway_overlap_genes,
            }
        )
        if sanity:
            return {"background": {}}, [0.1, 0.2], 3.0, 4.0
        return {"objects": {}}

    monkeypatch.setattr(analysis, "load", fake_load)

    result = analysis.main_analysis(
        perturbed_tracks="all",
        overall_perturbation_analysis=True,
        score="avg_backScore",
        items=None,
    )

    assert result == {"ok": True}
    assert any(call["sanity"] for call in load_calls)
    object_call = next(call for call in load_calls if call["sanity"] is False)
    assert object_call["mean_distance_change"] == pytest.approx(3.0)
    assert object_call["std_distance_change"] == pytest.approx(4.0)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_grouped_track_weight_uses_stage_normalized_average_weights(analysis_cls):
    analysis = analysis_cls(
        _make_track_weight_adata(),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
    )
    tracks = ["0-0-1-0", "0-14-15-10", "0-14-15-18"]

    weights = analysis.get_track_weight(tracks, ["0"])

    assert weights["0-0-1-0"] == pytest.approx(0.6095959595959596)
    assert weights["0-14-15-10"] == pytest.approx(0.20656565656565656)
    assert weights["0-14-15-18"] == pytest.approx(0.18383838383838383)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_grouped_track_weight_returns_one_for_single_selected_trajectory(analysis_cls):
    analysis = analysis_cls(
        _make_track_weight_adata(),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
    )

    weights = analysis.get_track_weight(["0-0-1-0"], ["0"])

    assert list(weights.keys()) == ["0-0-1-0"]
    assert weights["0-0-1-0"] == pytest.approx(1.0)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_track_weight_preserves_legacy_weighting_for_all_tracks(analysis_cls):
    analysis = analysis_cls(
        _make_track_weight_adata(),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
    )
    tracks = ["0-0-1-0", "0-14-15-10", "0-14-15-18"]

    weights = analysis.get_track_weight(tracks, "all")

    assert weights["0-0-1-0"] == pytest.approx(27 / 33)
    assert weights["0-14-15-10"] == pytest.approx(11 / 33)
    assert weights["0-14-15-18"] == pytest.approx(10 / 33)


@pytest.mark.parametrize("analysis_cls", [StandardAnalysis, NormalizedAnalysis])
def test_get_track_percentage_alias_matches_get_track_weight(analysis_cls):
    analysis = analysis_cls(
        _make_track_weight_adata(),
        target_directory=".",
        log2fc=0.5,
        mode="pathway",
    )
    tracks = ["0-0-1-0", "0-14-15-10", "0-14-15-18"]

    assert analysis.get_track_percentage(tracks, ["0"]) == analysis.get_track_weight(tracks, ["0"])
