from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
import pickle

import pandas as pd
import pytest

import UNAGI.monitoring.custom_perturbation as custom_perturbation_module
import UNAGI.perturbations.exact_parallel_executor as exact_parallel_module

from UNAGI.monitoring.custom_perturbation import (
    BACKEND_MODE_BY_MODE,
    PerturbationContext,
    PerturbationContextCache,
    _render_ranked_plot,
    autocomplete_suggestions,
    validate_custom_perturbation_payload,
)

from .interactive_analysis_test_helpers import write_analysis_dataset


def _prepare_analysis_dir(tmp_path: Path) -> Path:
    analysis_dir = tmp_path / "analysis_results"
    write_analysis_dataset(analysis_dir)
    model_save_dir = analysis_dir / "model_save"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    (model_save_dir / "training_parameters.json").write_text(
        json.dumps({"task": "example_task"}),
        encoding="utf-8",
    )
    (model_save_dir / "example_task_1.pth").write_text("stub", encoding="utf-8")
    (analysis_dir / "idrem").mkdir(parents=True, exist_ok=True)
    return analysis_dir


def test_validate_custom_perturbation_payload_canonicalizes_signed_genes_and_resolves_auto_mode(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "gene",
            "scope_kind": "track",
            "track_key": "0-2-4",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "signed_gene_text": "gene0:+, GENE1:-",
        },
        gpu_available=True,
    )

    assert result["ok"] is True
    assert result["normalized"]["target_spec"]["signed_genes"] == ["GENE0:+", "GENE1:-"]
    assert result["normalized"]["execution_mode"] == "cell_exact"
    assert result["normalized"]["resolved_execution_backend"] == "gpu_exact"
    assert result["normalized"]["track_label"] == "track 0 AT1"


def test_validate_custom_perturbation_payload_defaults_plain_genes_to_positive_direction(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "gene",
            "scope_kind": "overall",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "gene_text": "gene0",
        },
        gpu_available=True,
    )

    assert result["ok"] is True
    assert result["normalized"]["target_spec"]["signed_genes"] == ["GENE0:+"]


def test_validate_custom_perturbation_payload_accepts_suffix_signed_genes_without_colon(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "gene",
            "scope_kind": "overall",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "signed_gene_text": "gene0+, gene1-",
        },
        gpu_available=True,
    )

    assert result["ok"] is True
    assert result["normalized"]["target_spec"]["signed_genes"] == ["GENE0:+", "GENE1:-"]


def test_validate_custom_perturbation_payload_accepts_mixed_signed_gene_formats_for_drug_mode(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "drug",
            "scope_kind": "overall",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "target_name": "DrugA",
            "signed_gene_text": "gene0+, gene1:-",
        },
        gpu_available=True,
    )

    assert result["ok"] is True
    assert result["normalized"]["target_spec"]["signed_genes"] == ["GENE0:+", "GENE1:-"]


def test_validate_custom_perturbation_payload_resolves_cpu_exact_parallel_when_gpu_is_unavailable(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    monkeypatch.setattr(
        "UNAGI.monitoring.custom_perturbation._track_scope_cell_count",
        lambda adata, raw_track: 400_000,
    )

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "gene",
            "scope_kind": "track",
            "track_key": "0-2-4",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "cell_exact",
            "gene_text": "GENE0",
        },
        gpu_available=False,
    )

    assert result["normalized"]["execution_mode"] == "cell_exact"
    assert result["normalized"]["resolved_execution_backend"] == "cpu_exact_parallel"
    assert result["normalized"]["cpu_workers"] >= 1


def test_validate_custom_perturbation_payload_resolves_cpu_centroid_when_requested(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "gene",
            "scope_kind": "overall",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "centroid",
            "gene_text": "GENE0",
        },
        gpu_available=False,
    )

    assert result["normalized"]["execution_mode"] == "centroid"
    assert result["normalized"]["resolved_execution_backend"] == "cpu_centroid"

def test_validate_custom_perturbation_payload_accepts_stage_scope(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    result = validate_custom_perturbation_payload(
        analysis_dir=analysis_dir,
        analysis_job_id="analyze_test",
        analysis_iteration=1,
        raw_payload={
            "mode": "pathway",
            "scope_kind": "stage",
            "stage_key": "1",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "target_name": "PathA",
            "gene_text": "gene0, gene1",
        },
        gpu_available=True,
    )

    assert result["ok"] is True
    assert result["normalized"]["stage_key"] == "1"
    assert result["normalized"]["stage_label"] == "stage 1"
    assert result["normalized"]["stage_index"] == 1
def test_perturbation_context_cache_reuses_hot_context(tmp_path):
    skill_root = tmp_path / "skill_root"
    analysis_dir = _prepare_analysis_dir(tmp_path)
    cache = PerturbationContextCache(skill_root=skill_root)
    payload = {
        "analysis_dir": str(analysis_dir),
        "analysis_iteration": 1,
    }

    context_one = cache.get(payload)
    context_two = cache.get(payload)

    assert context_one is context_two
    assert context_one.model_path.name == "example_task_1.pth"


def test_perturbation_context_uses_correct_runner_constructor_per_execution_mode(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    calls: list[tuple[str, tuple, dict]] = []

    class _ExactRunner:
        def __init__(self, *args, **kwargs):
            calls.append(("exact", args, kwargs))

    class _CentroidRunner:
        def __init__(self, *args, **kwargs):
            calls.append(("centroid", args, kwargs))

    monkeypatch.setattr(custom_perturbation_module, "ExactPerturbation", _ExactRunner)
    monkeypatch.setattr(custom_perturbation_module, "CentroidPerturbation", _CentroidRunner)

    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    context.get_runner("cell_exact")
    context.get_runner("centroid")

    assert calls[0][0] == "exact"
    assert calls[0][2]["config_path"] == analysis_dir / "model_save" / "training_parameters.json"
    assert calls[1][0] == "centroid"
    assert calls[1][2] == {}


def test_perturbation_context_uses_cpu_override_for_cpu_exact_parallel(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    calls: list[tuple[tuple, dict]] = []

    class _ExactRunner:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(custom_perturbation_module, "ExactPerturbation", _ExactRunner)

    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    context.get_runner("cpu_exact_parallel")

    assert calls[0][1]["device_override"] == "cpu"


def test_perturbation_context_uses_gpu_override_for_gpu_exact(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    calls: list[tuple[tuple, dict]] = []

    class _ExactRunner:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(custom_perturbation_module, "ExactPerturbation", _ExactRunner)

    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    context.get_runner("gpu_exact")

    assert calls[0][1]["device_override"] == "cuda:0"


def test_exact_parallel_executor_matches_serial_shards(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    dataset_path = analysis_dir / "dataset.h5ad"
    attribute_path = analysis_dir / "attribute.pkl"
    model_path = analysis_dir / "model_save" / "example_task_1.pth"
    training_params_path = analysis_dir / "model_save" / "training_parameters.json"
    idrem_dir = analysis_dir / "idrem"

    class _ExactRunner:
        def __init__(self, adata, *_args, **_kwargs):
            self.adata = adata

        def run(self, mode, perturb_change, **_kwargs):
            items = self.adata.uns["data_drug_overlap_genes"]
            self.adata.uns["drug_perturbation_deltaD"] = {
                str(perturb_change): {
                    "overall": {
                        name: float(len(genes))
                        for name, genes in items.items()
                    }
                }
            }

    monkeypatch.setattr(exact_parallel_module, "ExactPerturbation", _ExactRunner)

    items = {
        "DrugA": ["GENE0:+", "GENE1:-"],
        "DrugB": ["GENE0:+", "GENE2:-"],
        "DrugC": ["GENE1:+", "GENE2:-"],
        "DrugD": ["GENE0:+", "GENE3:-"],
    }
    serial = exact_parallel_module.run_exact_parallel_delta(
        dataset_path=dataset_path,
        attribute_path=attribute_path,
        model_path=model_path,
        training_params_path=training_params_path,
        idrem_dir=idrem_dir,
        backend_mode="drug",
        items=items,
        perturb_change=0.5,
        direction_mode="two_sided",
        cpu_workers=1,
    )
    parallel = exact_parallel_module.run_exact_parallel_delta(
        dataset_path=dataset_path,
        attribute_path=attribute_path,
        model_path=model_path,
        training_params_path=training_params_path,
        idrem_dir=idrem_dir,
        backend_mode="drug",
        items=items,
        perturb_change=0.5,
        direction_mode="two_sided",
        cpu_workers=2,
    )

    assert parallel == serial


def test_autocomplete_suggestions_match_genes_pathways_and_drugs(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)

    gene_suggestions = autocomplete_suggestions(analysis_dir, kind="gene", query="gene0")
    assert gene_suggestions[0]["value"] == "GENE0"

    pathway_suggestions = autocomplete_suggestions(analysis_dir, kind="pathway", query="patha")
    assert pathway_suggestions[0]["value"] == "PathA"
    assert pathway_suggestions[0]["fill_gene_text"] == "GENE0, GENE1"

    drug_suggestions = autocomplete_suggestions(analysis_dir, kind="drug", query="druga")
    assert drug_suggestions[0]["value"] == "DrugA"
    assert drug_suggestions[0]["fill_signed_gene_text"] == "GENE0:+, GENE1:-"


def test_gene_set_custom_result_table_uses_drug_backend_and_filters_internal_name(tmp_path, monkeypatch):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    adata.uns["drug_perturbation_score"] = {"0.5": {"overall": {"top_compounds": {"0": "custom_gene_set_test"}}}}

    monkeypatch.setattr(
        "UNAGI.plotting.manuscript_figures.extract_perturbation_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "compound": ["custom_gene_set_test", "bg_0"],
                "perturbation score": [0.5, 0.4],
                "pval_adjusted": [0.01, 0.02],
            }
        ),
    )

    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )
    runner = SimpleNamespace(adata=adata)
    table, resolved_track_key = context._extract_result_table(
        runner,
        request_payload={
            "mode": "gene",
            "perturb_change": 0.5,
            "scope_kind": "overall",
            "target_spec": {
                "signed_genes": ["GENE1:+", "GENE0:-"],
                "internal_name": "custom_gene_set_test",
            },
        },
        backend_mode="drug",
    )

    assert BACKEND_MODE_BY_MODE["gene"] == "drug"
    assert resolved_track_key == "overall"
    assert table["gene set"].tolist() == ["GENE1:+, GENE0:-"]


def test_render_ranked_plot_accepts_gene_set_column():
    table = pd.DataFrame(
        {
            "gene set": ["GENE0:+, GENE1:-"],
            "perturbation score": [0.42],
            "pval_adjusted": [0.01],
        }
    )

    png_bytes, pdf_bytes = _render_ranked_plot(table, mode="gene")

    assert png_bytes.startswith(b"\x89PNG")
    assert pdf_bytes.startswith(b"%PDF")


def test_load_result_bundle_regenerates_plot_from_score_when_pvalues_are_missing(tmp_path):
    result_dir = tmp_path / "result_bundle"
    result_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "gene set": ["GENE0:+, GENE1:-"],
            "perturbation score": [0.42],
            "pval_adjusted": [float("nan")],
        }
    )
    table_csv_path = result_dir / "result_table.csv"
    table_json_path = result_dir / "result_table.json"
    plot_png_path = result_dir / "ranked_bar.png"
    plot_pdf_path = result_dir / "ranked_bar.pdf"
    summary_path = result_dir / "summary.json"
    table.to_csv(table_csv_path, index=False)
    table_json_path.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    plot_png_path.write_bytes(b"old")
    plot_pdf_path.write_bytes(b"old")
    summary_path.write_text(
        json.dumps(
            {
                "request": {"mode": "gene"},
                "metadata": {},
                "table_csv_path": str(table_csv_path),
                "table_json_path": str(table_json_path),
                "plot_png_path": str(plot_png_path),
                "plot_pdf_path": str(plot_pdf_path),
                "preview": {"columns": ["gene set", "perturbation score", "pval_adjusted"], "rows": [["GENE0:+, GENE1:-", 0.42, None]]},
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )

    bundle = custom_perturbation_module.load_result_bundle(result_dir)

    assert bundle["metadata"]["plot_basis"] == "score"
    assert plot_png_path.read_bytes().startswith(b"\x89PNG")
    assert plot_pdf_path.read_bytes().startswith(b"%PDF")


def test_background_cache_builds_100_random_items_per_signature_size(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    adata.uns["data_drug_overlap_genes"] = {}
    adata.uns["random_background_perturbation_deltaD"] = {}
    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    captured_items = {}

    class _Runner:
        def __init__(self, adata):
            self.adata = adata
            self.calls = []

        def run(self, mode, perturb_change, **kwargs):
            self.calls.append((mode, perturb_change, kwargs))
            payload = {"0.5": {"overall": {}}}
            for key in captured_items:
                payload["0.5"]["overall"][key] = float(len(key))
            self.adata.uns["random_background_perturbation_deltaD"] = payload

    runner = _Runner(adata)

    original_set_backend_items = context._set_backend_items

    def _capture_items(runner_obj, *, mode, items):
        captured_items.clear()
        captured_items.update(items)
        original_set_backend_items(runner_obj, mode=mode, items=items)

    context._set_backend_items = _capture_items  # type: ignore[method-assign]
    background = context._load_or_build_background(
        runner=runner,
        backend_mode="drug",
        execution_mode="centroid",
        direction_mode="two_sided",
        perturb_change=0.5,
        sizes=[2],
        use_cuda=False,
        device="cpu",
        log=lambda _message: None,
    )

    assert len(captured_items) == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES
    assert runner.calls[0][0] == "random_drug_background"
    assert runner.calls[0][2]["random_times"] == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES
    assert len(background["0.5"]["overall"]) == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES


def test_background_cache_reuses_precomputed_drug_background_when_size_matches(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    adata.uns["data_drug_overlap_genes"] = {f"Drug{i}": ["GENE0", "GENE1"] for i in range(120)}
    adata.uns["random_background_perturbation_deltaD"] = {
        "0.5": {"overall": {f"Drug{i}": float(i) for i in range(120)}},
        "2.0": {"overall": {f"Drug{i}": float(i) for i in range(120)}},
    }
    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    class _Runner:
        def __init__(self, adata):
            self.adata = adata
            self.calls = []

        def run(self, mode, perturb_change, **kwargs):
            self.calls.append((mode, perturb_change, kwargs))

    runner = _Runner(adata)
    background = context._load_or_build_background(
        runner=runner,
        backend_mode="drug",
        execution_mode="centroid",
        direction_mode="two_sided",
        perturb_change=0.5,
        sizes=[2],
        use_cuda=False,
        device="cpu",
        log=lambda _message: None,
    )

    assert runner.calls == []
    assert len(background["0.5"]["overall"]) == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES
    assert len(background["2.0"]["overall"]) == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES


def test_background_cache_does_not_reuse_two_sided_background_for_one_sided_requests(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    adata.uns["analysis_metadata"] = {
        "direction_mode": "two_sided",
        "background_direction_mode": "two_sided",
        "background_perturb_change": 0.5,
    }
    adata.uns["data_drug_overlap_genes"] = {f"Drug{i}": ["GENE0", "GENE1"] for i in range(120)}
    adata.uns["random_background_perturbation_deltaD"] = {
        "0.5": {"overall": {f"Drug{i}": float(i) for i in range(120)}},
        "2.0": {"overall": {f"Drug{i}": float(i) for i in range(120)}},
    }
    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    class _Runner:
        def __init__(self, adata):
            self.adata = adata
            self.calls = []

        def run(self, mode, perturb_change, **kwargs):
            self.calls.append((mode, perturb_change, kwargs))
            self.adata.uns["random_background_perturbation_deltaD"] = {
                "0.5": {"overall": {f"bg_{index}": float(index) for index in range(custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES)}}
            }

    runner = _Runner(adata)
    background = context._load_or_build_background(
        runner=runner,
        backend_mode="drug",
        execution_mode="centroid",
        direction_mode="one_sided",
        perturb_change=0.5,
        sizes=[2],
        use_cuda=False,
        device="cpu",
        log=lambda _message: None,
    )

    assert runner.calls and runner.calls[0][0] == "random_drug_background"
    assert "2.0" not in background
    assert len(background["0.5"]["overall"]) == custom_perturbation_module.DEFAULT_BACKGROUND_RANDOM_TIMES


def test_build_form_context_reports_saved_background_direction_mode(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _dataset_path, attribute_path = custom_perturbation_module._read_analysis_adata(analysis_dir)
    adata.uns["analysis_metadata"] = {
        "direction_mode": "two_sided",
        "background_direction_mode": "two_sided",
        "background_perturb_change": 0.5,
    }
    adata.uns["random_background_direction_mode"] = "two_sided"
    adata.uns["random_background_perturb_change"] = 0.5
    adata.uns["analysis_direction_mode"] = "two_sided"
    with attribute_path.open("wb") as handle:
        pickle.dump(dict(adata.uns), handle)

    form = custom_perturbation_module.build_form_context(analysis_dir)

    assert form["default_direction_mode"] == "two_sided"
    assert form["background_direction_mode"] == "two_sided"
    assert "different random backgrounds" in form["background_note"]
    assert {"value": "stage", "label": "stage"} in form["scope_options"]
    assert form["stage_options"][1]["value"] == "1"


def test_run_request_requires_valid_pvalues(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    class _Runner:
        def __init__(self, adata):
            self.adata = adata

        def run(self, *_args, **_kwargs):
            return None

        def analysis(self, *_args, **_kwargs):
            return None

    runner = _Runner(adata)
    context.get_runner = lambda _execution_mode: runner  # type: ignore[method-assign]
    context._load_or_build_background = lambda **_kwargs: {"0.5": {"overall": {"bg_0": 0.1}}}  # type: ignore[method-assign]
    context._set_backend_items = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    context._runner_clear_keys = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    context._extract_result_table = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        pd.DataFrame({"gene set": ["GENE0:+, GENE1:-"], "perturbation score": [0.42], "pval_adjusted": [float("nan")]}),
        "overall",
    )

    with pytest.raises(ValueError, match="did not produce valid p-values"):
        context.run_request(
            {
                "mode": "gene",
                "execution_mode": "centroid",
                "perturb_change": 0.5,
                "direction_mode": "two_sided",
                "scope_kind": "overall",
                "target_spec": {"signed_genes": ["GENE0:+", "GENE1:-"], "internal_name": "custom_gene_set_test"},
                "analysis_iteration": 1,
            },
            result_dir=tmp_path / "result_bundle",
            log=lambda _message: None,
        )


def test_run_request_passes_stage_scope_to_runner_analysis(tmp_path):
    analysis_dir = _prepare_analysis_dir(tmp_path)
    adata, _, _ = custom_perturbation_module._read_analysis_adata(analysis_dir)
    context = PerturbationContext(
        skill_root=tmp_path / "skill_root",
        analysis_dir=analysis_dir,
        adata=adata,
        metadata={},
        model_path=analysis_dir / "model_save" / "example_task_1.pth",
        training_params_path=analysis_dir / "model_save" / "training_parameters.json",
        cache_root=tmp_path / "cache",
        analysis_iteration=1,
    )

    captured: dict[str, object] = {}

    class _Runner:
        def __init__(self, adata):
            self.adata = adata
            self.last_gpu_batch_meta = {}

        def run(self, *_args, **_kwargs):
            return None

        def analysis(self, mode, perturb_change, **kwargs):
            captured["mode"] = mode
            captured["perturb_change"] = perturb_change
            captured["kwargs"] = dict(kwargs)
            self.adata.uns["pathway_perturbation_score"] = {
                "0.5": {"overall": {"top_compounds": {"0": "PathA"}}}
            }

    runner = _Runner(adata)
    context.get_runner = lambda _execution_mode: runner  # type: ignore[method-assign]
    context._load_or_build_background = lambda **_kwargs: {"0.5": {"overall": {"bg_0": 0.1}}}  # type: ignore[method-assign]
    context._set_backend_items = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    context._runner_clear_keys = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    context._extract_result_table = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        pd.DataFrame({"pathways": ["PathA"], "perturbation score": [0.42], "pval_adjusted": [0.01]}),
        "overall",
    )

    summary = context.run_request(
        {
            "mode": "pathway",
            "execution_mode": "centroid",
            "resolved_execution_backend": "gpu_centroid",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "scope_kind": "stage",
            "stage_key": "1",
            "stage_label": "stage 1",
            "stage_index": 1,
            "target_spec": {"name": "PathA", "genes": ["GENE0", "GENE1"]},
            "analysis_iteration": 1,
            "dataset_path": str(analysis_dir / "dataset.h5ad"),
            "attribute_path": str(analysis_dir / "attribute.pkl"),
            "cpu_workers": 1,
        },
        result_dir=tmp_path / "result_bundle",
        log=lambda _message: None,
    )

    assert captured["mode"] == "pathway"
    assert captured["kwargs"]["stage"] == 1
    assert summary["metadata"]["stage_label"] == "stage 1"
