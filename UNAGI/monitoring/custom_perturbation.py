from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from UNAGI.plotting import plot_perturbation_ranked_bar
from UNAGI.perturbations.direction_utils import infer_direction_mode
from UNAGI.perturbations.exact_parallel_executor import (
    default_cpu_workers,
    merge_directional_payloads,
    run_exact_parallel_delta,
)
from UNAGI.perturbations.perturbation import perturbation as ExactPerturbation
from UNAGI.perturbations.perturbation_centroid import perturbation as CentroidPerturbation

from .interactive_analysis import (
    _extract_metadata,
    _preview_frame,
    _resolve_gene_list,
    _track_cluster_id,
)
from .visualization_pipeline import _read_analysis_adata

GENE_LIMIT = 50
DEFAULT_BACKGROUND_RANDOM_TIMES = 100
BACKGROUND_CACHE_VERSION = 3
PLOT_TOP_N = 20
AUTOCOMPLETE_LIMIT = 12
AUTOCOMPLETE_CACHE_SIZE = 4
FORM_CONTEXT_CACHE_SIZE = 4
METADATA_CACHE_FILE = "custom_perturbation_metadata.json"

SUPPORTED_MODES = ("gene", "pathway", "drug")
SUPPORTED_SCOPE_KINDS = ("overall", "track")
SUPPORTED_EXECUTION_MODES = ("auto", "centroid", "cell_exact")
SUPPORTED_DIRECTION_MODES = ("one_sided", "two_sided")
SUPPORTED_RESOLVED_BACKENDS = ("gpu_exact", "gpu_centroid", "cpu_exact_parallel", "cpu_centroid")

BACKEND_MODE_BY_MODE = {
    "gene": "drug",
    "pathway": "pathway",
    "drug": "drug",
}

TABLE_ITEM_COLUMNS = {
    "gene": "gene",
    "pathway": "pathways",
    "drug": "compound",
}

RESULT_FILE_NAMES = {
    "request": "request.json",
    "summary": "summary.json",
    "table_csv": "result_table.csv",
    "table_json": "result_table.json",
    "plot_png": "ranked_bar.png",
    "plot_pdf": "ranked_bar.pdf",
}

_AUTOCOMPLETE_CATALOG_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_AUTOCOMPLETE_CATALOG_LOCK = threading.Lock()
_FORM_CONTEXT_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_FORM_CONTEXT_LOCK = threading.Lock()


def _detect_gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sanitize_queue_key(queue_key: str) -> str:
    return str(queue_key).replace(":", "_").replace("/", "_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_identity(analysis_dir: Path) -> tuple[Any, ...]:
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    dataset_path = analysis_dir / "dataset.h5ad"
    attribute_path = analysis_dir / "attribute.pkl"
    model_save_dir = analysis_dir / "model_save"
    training_params_path = model_save_dir / "training_parameters.json"
    identity = [str(analysis_dir)]
    for path in (dataset_path, attribute_path, training_params_path):
        if not path.exists():
            identity.append((str(path), None, None))
            continue
        stat = path.stat()
        identity.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(identity)


def _analysis_hash(analysis_dir: Path) -> str:
    payload = json.dumps(_analysis_identity(analysis_dir), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _normalize_direction_mode(raw: Any) -> str:
    value = str(raw or "two_sided").strip().lower()
    if value not in SUPPORTED_DIRECTION_MODES:
        raise ValueError("direction_mode must be one of: one_sided, two_sided")
    return value


def _extract_analysis_direction_metadata(adata, *, perturb_change: float | None = None) -> dict[str, Any]:
    uns = getattr(adata, "uns", {}) or {}
    metadata: dict[str, Any] = {}
    analysis_meta = uns.get("analysis_metadata")
    if isinstance(analysis_meta, dict):
        metadata.update(analysis_meta)

    analysis_direction = metadata.get("direction_mode") or uns.get("analysis_direction_mode")
    if analysis_direction in SUPPORTED_DIRECTION_MODES:
        metadata["direction_mode"] = _normalize_direction_mode(analysis_direction)

    background_direction = (
        metadata.get("background_direction_mode")
        or uns.get("perturbation_background_direction_mode")
        or uns.get("random_background_direction_mode")
    )
    if background_direction in SUPPORTED_DIRECTION_MODES:
        metadata["background_direction_mode"] = _normalize_direction_mode(background_direction)

    metadata["background_perturb_change"] = (
        metadata.get("background_perturb_change")
        or uns.get("perturbation_background_perturb_change")
        or uns.get("random_background_perturb_change")
    )

    if "background_direction_mode" not in metadata:
        source_background = uns.get("random_background_perturbation_deltaD", {})
        if isinstance(source_background, dict) and source_background:
            try:
                bound = float(perturb_change or metadata.get("background_perturb_change") or 0.5)
                metadata["background_direction_mode"] = infer_direction_mode(source_background, bound)
            except Exception:
                pass
    return metadata


def _normalize_scope_kind(raw: Any) -> str:
    value = str(raw or "overall").strip().lower()
    if value not in SUPPORTED_SCOPE_KINDS:
        raise ValueError("scope_kind must be one of: overall, track")
    return value


def _normalize_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of: {', '.join(SUPPORTED_MODES)}")
    return value


def _normalize_execution_mode(raw: Any) -> str:
    value = str(raw or "auto").strip().lower()
    if value not in SUPPORTED_EXECUTION_MODES:
        raise ValueError("execution_mode must be one of: auto, centroid, cell_exact")
    return value


def _normalize_cpu_workers(raw: Any) -> int:
    return default_cpu_workers(raw)


def _split_gene_text(raw_text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in str(raw_text or "").replace("\n", ",").split(","):
        for token in chunk.strip().split():
            token = token.strip()
            if token:
                tokens.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        folded = token.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        deduped.append(token)
    return deduped


def _parse_signed_gene_text(raw_text: str) -> list[str]:
    tokens = _split_gene_text(raw_text)
    normalized: list[str] = []
    for token in tokens:
        gene = token.strip()
        sign = ""
        if ":" in gene:
            gene, _sep, sign = gene.partition(":")
            gene = gene.strip()
            sign = sign.strip()
        elif gene.endswith(("+", "-")) and len(gene) > 1:
            sign = gene[-1]
            gene = gene[:-1].strip()
        if not gene or sign not in {"+", "-"}:
            raise ValueError("Drug signatures must use GENE:+ or GENE:- format.")
        normalized.append(f"{gene}:{sign}")
    return normalized


def _parse_optional_signed_gene_text(raw_text: str) -> list[str]:
    tokens = _split_gene_text(raw_text)
    normalized: list[str] = []
    for token in tokens:
        gene = token.strip()
        sign = ""
        sep = ""
        if ":" in gene:
            gene, sep, sign = gene.partition(":")
            gene = gene.strip()
            sign = sign.strip()
        elif gene.endswith(("+", "-")) and len(gene) > 1:
            sign = gene[-1]
            gene = gene[:-1].strip()
        if not gene:
            continue
        if not sign and not sep:
            normalized.append(f"{gene}:+")
            continue
        if sign not in {"+", "-"}:
            raise ValueError("Gene signatures must use GENE:+ or GENE:- format.")
        normalized.append(f"{gene}:{sign}")
    return normalized


def _bh_adjust(pvalues: Sequence[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=float))
    ordered = np.asarray(pvalues, dtype=float)[order]
    adjusted = np.empty(len(ordered), dtype=float)
    total = float(len(ordered))
    running = 1.0
    for reverse_index, pvalue in enumerate(ordered[::-1], start=1):
        rank = total - reverse_index + 1.0
        candidate = min(running, float(pvalue) * total / rank)
        running = candidate
        adjusted[len(ordered) - reverse_index] = min(1.0, candidate)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored.tolist()


def _table_item_column(mode: str) -> str:
    return TABLE_ITEM_COLUMNS[mode]


def _custom_gene_set_internal_name(genes: Sequence[str]) -> str:
    joined = ",".join(str(gene) for gene in genes)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"custom_gene_set_{digest}"


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        folded = token.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        deduped.append(token)
    return deduped


def _extract_ranked_names(score_payload: Any) -> list[str]:
    if not isinstance(score_payload, dict):
        return []
    names: list[str] = []
    for change_bucket in score_payload.values():
        if not isinstance(change_bucket, dict):
            continue
        for track_bucket in change_bucket.values():
            if not isinstance(track_bucket, dict):
                continue
            ranked = track_bucket.get("top_compounds", track_bucket)
            if isinstance(ranked, pd.DataFrame):
                for column in ("compound", "pathways", "gene"):
                    if column in ranked.columns:
                        names.extend(ranked[column].tolist())
                        break
                continue
            if isinstance(ranked, dict):
                for column in ("compound", "pathways", "gene"):
                    values = ranked.get(column)
                    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                        names.extend(list(values))
                        break
    return _dedupe_strings(names)


def _autocomplete_catalog_pathways(adata) -> dict[str, list[str]]:
    raw_items = adata.uns.get("data_pathway_overlap_genes")
    if isinstance(raw_items, dict) and raw_items:
        return {
            str(name): _dedupe_strings(genes)
            for name, genes in raw_items.items()
            if str(name or "").strip()
        }
    return {name: [] for name in _extract_ranked_names(adata.uns.get("pathway_perturbation_score"))}


def _autocomplete_catalog_drugs(adata) -> dict[str, list[str]]:
    raw_items = adata.uns.get("data_drug_overlap_genes")
    if isinstance(raw_items, dict) and raw_items:
        return {
            str(name): _dedupe_strings(genes)
            for name, genes in raw_items.items()
            if str(name or "").strip()
        }
    return {name: [] for name in _extract_ranked_names(adata.uns.get("drug_perturbation_score"))}


def _metadata_cache_path(analysis_dir: str | Path) -> Path:
    return Path(analysis_dir).expanduser().resolve() / METADATA_CACHE_FILE


def _build_form_context_from_adata(adata) -> dict[str, Any]:
    track_options = _extract_track_options(adata)
    analysis_direction = _extract_analysis_direction_metadata(adata)
    background_direction_mode = str(analysis_direction.get("background_direction_mode") or "")
    default_direction_mode = str(
        analysis_direction.get("direction_mode")
        or background_direction_mode
        or "two_sided"
    )
    background_direction_mode = background_direction_mode or default_direction_mode
    background_note = (
        f"Saved analysis background: {background_direction_mode}. "
        "One-sided and two-sided perturbations use different random backgrounds."
    )
    return {
        "mode_options": [
            {"value": "gene", "label": "Gene"},
            {"value": "pathway", "label": "Pathway"},
            {"value": "drug", "label": "Drug"},
        ],
        "scope_options": [
            {"value": "overall", "label": "overall"},
            {"value": "track", "label": "temporal track"},
        ],
        "execution_mode_options": [
            {"value": "auto", "label": "auto"},
            {"value": "centroid", "label": "centroid"},
            {"value": "cell_exact", "label": "cell_exact"},
        ],
        "direction_mode_options": [
            {"value": "two_sided", "label": "two_sided"},
            {"value": "one_sided", "label": "one_sided"},
        ],
        "track_options": track_options,
        "gene_limit": GENE_LIMIT,
        "default_direction_mode": default_direction_mode,
        "background_direction_mode": background_direction_mode,
        "background_note": background_note,
    }


def _build_metadata_cache_payload(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    adata, _dataset_path, _attribute_path = _read_analysis_adata(analysis_path)
    return {
        "analysis_hash": _analysis_hash(analysis_path),
        "catalog": {
            "genes": _dedupe_strings(adata.var_names.astype(str).tolist()),
            "pathway_items": _autocomplete_catalog_pathways(adata),
            "drug_items": _autocomplete_catalog_drugs(adata),
        },
        "form_context": _build_form_context_from_adata(adata),
    }


def _load_prewarmed_metadata_cache(analysis_dir: str | Path) -> dict[str, Any] | None:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    cache_path = _metadata_cache_path(analysis_path)
    if not cache_path.exists():
        return None
    try:
        payload = _read_json(cache_path)
    except Exception:
        return None
    if str(payload.get("analysis_hash") or "") != _analysis_hash(analysis_path):
        return None
    if not isinstance(payload.get("catalog"), dict) or not isinstance(payload.get("form_context"), dict):
        return None
    return payload


def prewarm_metadata_cache(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    payload = _build_metadata_cache_payload(analysis_path)
    _write_json(_metadata_cache_path(analysis_path), payload)
    cache_key = _analysis_hash(analysis_path)
    with _AUTOCOMPLETE_CATALOG_LOCK:
        _AUTOCOMPLETE_CATALOG_CACHE[cache_key] = copy.deepcopy(payload["catalog"])
        _AUTOCOMPLETE_CATALOG_CACHE.move_to_end(cache_key)
        while len(_AUTOCOMPLETE_CATALOG_CACHE) > AUTOCOMPLETE_CACHE_SIZE:
            _AUTOCOMPLETE_CATALOG_CACHE.popitem(last=False)
    with _FORM_CONTEXT_LOCK:
        _FORM_CONTEXT_CACHE[cache_key] = copy.deepcopy(payload["form_context"])
        _FORM_CONTEXT_CACHE.move_to_end(cache_key)
        while len(_FORM_CONTEXT_CACHE) > FORM_CONTEXT_CACHE_SIZE:
            _FORM_CONTEXT_CACHE.popitem(last=False)
    return payload


def _build_autocomplete_catalog(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    adata, _dataset_path, _attribute_path = _read_analysis_adata(analysis_path)
    return {
        "genes": _dedupe_strings(adata.var_names.astype(str).tolist()),
        "pathway_items": _autocomplete_catalog_pathways(adata),
        "drug_items": _autocomplete_catalog_drugs(adata),
    }


def _get_autocomplete_catalog(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    cache_key = _analysis_hash(analysis_path)
    with _AUTOCOMPLETE_CATALOG_LOCK:
        cached = _AUTOCOMPLETE_CATALOG_CACHE.get(cache_key)
        if cached is not None:
            _AUTOCOMPLETE_CATALOG_CACHE.move_to_end(cache_key)
            return cached
    prewarmed = _load_prewarmed_metadata_cache(analysis_path)
    if prewarmed is not None:
        catalog = dict(prewarmed["catalog"])
    else:
        catalog = _build_autocomplete_catalog(analysis_path)
    with _AUTOCOMPLETE_CATALOG_LOCK:
        _AUTOCOMPLETE_CATALOG_CACHE[cache_key] = catalog
        _AUTOCOMPLETE_CATALOG_CACHE.move_to_end(cache_key)
        while len(_AUTOCOMPLETE_CATALOG_CACHE) > AUTOCOMPLETE_CACHE_SIZE:
            _AUTOCOMPLETE_CATALOG_CACHE.popitem(last=False)
    return catalog


def _match_rank(name: str, query: str) -> tuple[int, int, int, str] | None:
    folded_name = name.casefold()
    folded_query = query.casefold()
    if not folded_query:
        return (3, 0, len(name), folded_name)
    if folded_name == folded_query:
        return (0, 0, len(name), folded_name)
    if folded_name.startswith(folded_query):
        return (1, 0, len(name), folded_name)
    position = folded_name.find(folded_query)
    if position >= 0:
        return (2, position, len(name), folded_name)
    return None


def autocomplete_suggestions(
    analysis_dir: str | Path,
    *,
    kind: str,
    query: str = "",
    limit: int = AUTOCOMPLETE_LIMIT,
) -> list[dict[str, Any]]:
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind not in {"gene", "pathway", "drug"}:
        raise ValueError("kind must be one of: gene, pathway, drug")
    catalog = _get_autocomplete_catalog(analysis_dir)
    limit = max(1, min(int(limit or AUTOCOMPLETE_LIMIT), 50))
    names_source: list[str]
    items_by_name: dict[str, list[str]] = {}
    if normalized_kind == "gene":
        names_source = catalog["genes"]
    elif normalized_kind == "pathway":
        items_by_name = catalog["pathway_items"]
        names_source = list(items_by_name.keys())
    else:
        items_by_name = catalog["drug_items"]
        names_source = list(items_by_name.keys())

    matched = []
    for name in names_source:
        rank = _match_rank(name, str(query or "").strip())
        if rank is None:
            continue
        matched.append((rank, name))
    matched.sort(key=lambda item: item[0])

    suggestions: list[dict[str, Any]] = []
    for _rank, name in matched[:limit]:
        entry: dict[str, Any] = {"value": name, "label": name}
        if normalized_kind == "pathway":
            genes = items_by_name.get(name, [])
            entry["gene_count"] = len(genes)
            if genes:
                entry["fill_gene_text"] = ", ".join(genes)
        elif normalized_kind == "drug":
            signed_genes = items_by_name.get(name, [])
            entry["gene_count"] = len(signed_genes)
            if signed_genes:
                entry["fill_signed_gene_text"] = ", ".join(signed_genes)
        suggestions.append(entry)
    return suggestions


def _render_ranked_plot(table: pd.DataFrame, *, mode: str) -> tuple[bytes, bytes]:
    plot_table = table
    if "gene set" in plot_table.columns and not any(
        candidate in plot_table.columns for candidate in ("compound", "pathways", "gene", "gene_combination", "item")
    ):
        plot_table = plot_table.rename(columns={"gene set": "item"})
    effective_pval_col = None
    for candidate in ("pval_adjusted", "pval"):
        if candidate not in plot_table.columns:
            continue
        values = pd.to_numeric(plot_table[candidate], errors="coerce")
        if values.notna().any():
            effective_pval_col = candidate
            break
    figure, _axis, _work = plot_perturbation_ranked_bar(
        plot_table,
        top_n=min(PLOT_TOP_N, len(plot_table)),
        pval_col=effective_pval_col or "pval_adjusted",
        use_neglog10_pval=bool(effective_pval_col),
        show=False,
    )
    png_buffer = io.BytesIO()
    pdf_buffer = io.BytesIO()
    figure.savefig(png_buffer, format="png", dpi=160, bbox_inches="tight")
    figure.savefig(pdf_buffer, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return png_buffer.getvalue(), pdf_buffer.getvalue()


def _write_result_bundle(
    *,
    result_dir: Path,
    normalized_request: dict[str, Any],
    table: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    request_path = result_dir / RESULT_FILE_NAMES["request"]
    summary_path = result_dir / RESULT_FILE_NAMES["summary"]
    table_csv_path = result_dir / RESULT_FILE_NAMES["table_csv"]
    table_json_path = result_dir / RESULT_FILE_NAMES["table_json"]
    plot_png_path = result_dir / RESULT_FILE_NAMES["plot_png"]
    plot_pdf_path = result_dir / RESULT_FILE_NAMES["plot_pdf"]

    _write_json(request_path, normalized_request)
    table.to_csv(table_csv_path, index=False)
    table_json_path.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    png_bytes, pdf_bytes = _render_ranked_plot(table, mode=normalized_request["mode"])
    plot_png_path.write_bytes(png_bytes)
    plot_pdf_path.write_bytes(pdf_bytes)

    summary = {
        "request": normalized_request,
        "metadata": metadata,
        "table_csv_path": str(table_csv_path),
        "table_json_path": str(table_json_path),
        "plot_png_path": str(plot_png_path),
        "plot_pdf_path": str(plot_pdf_path),
        "preview": _preview_frame(table),
        "row_count": int(len(table)),
    }
    effective_pval_col = None
    for candidate in ("pval_adjusted", "pval"):
        if candidate not in table.columns:
            continue
        values = pd.to_numeric(table[candidate], errors="coerce")
        if values.notna().any():
            effective_pval_col = candidate
            break
    summary["metadata"]["plot_basis"] = "pvalue" if effective_pval_col else "score"
    _write_json(summary_path, summary)
    return summary


def load_result_bundle(result_dir: str | Path) -> dict[str, Any]:
    result_path = Path(result_dir).expanduser().resolve()
    summary_path = result_path / RESULT_FILE_NAMES["summary"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Custom perturbation summary not found: {summary_path}")
    summary = _read_json(summary_path)
    metadata = summary.setdefault("metadata", {})
    if "plot_basis" not in metadata:
        table_csv_path = Path(str(summary.get("table_csv_path") or "")).expanduser()
        plot_png_path = Path(str(summary.get("plot_png_path") or "")).expanduser()
        plot_pdf_path = Path(str(summary.get("plot_pdf_path") or "")).expanduser()
        if table_csv_path.exists():
            table = pd.read_csv(table_csv_path)
            png_bytes, pdf_bytes = _render_ranked_plot(
                table,
                mode=str((summary.get("request") or {}).get("mode") or "gene"),
            )
            plot_png_path.write_bytes(png_bytes)
            plot_pdf_path.write_bytes(pdf_bytes)
            effective_pval_col = None
            for candidate in ("pval_adjusted", "pval"):
                if candidate not in table.columns:
                    continue
                values = pd.to_numeric(table[candidate], errors="coerce")
                if values.notna().any():
                    effective_pval_col = candidate
                    break
            metadata["plot_basis"] = "pvalue" if effective_pval_col else "score"
            _write_json(summary_path, summary)
    return summary


def _pick_model_path(model_save_dir: Path, iteration: int) -> Path:
    training_params_path = model_save_dir / "training_parameters.json"
    if training_params_path.exists():
        params = json.loads(training_params_path.read_text(encoding="utf-8"))
        task = str(params.get("task", "")).strip()
        if task:
            candidate = model_save_dir / f"{task}_{int(iteration)}.pth"
            if candidate.exists():
                return candidate
    candidates = sorted(path for path in model_save_dir.glob("*.pth") if "_dis_" not in path.name)
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found in {model_save_dir}")
    return candidates[0]


def _track_scope_cell_count(adata, raw_track: str) -> int:
    clusters = [segment for segment in str(raw_track).split("-") if segment != ""]
    if not clusters:
        raise ValueError(f"Invalid track key: {raw_track!r}")
    total = 0
    for stage_index, cluster in enumerate(clusters):
        stage_mask = adata.obs["stage"].astype(str) == str(stage_index)
        cluster_mask = adata.obs["leiden"].astype(str) == str(cluster)
        total += int(np.count_nonzero(stage_mask & cluster_mask))
    return total


def _extract_track_options(adata) -> list[dict[str, str]]:
    metadata = _extract_metadata(adata)
    tracks = metadata["dynamic_markers"]["tracks"]
    return [{"value": str(item["value"]), "label": str(item["label"])} for item in tracks]


def _resolve_track_option(track_options: list[dict[str, str]], raw_track: str) -> dict[str, str]:
    for item in track_options:
        if item["value"] == raw_track:
            return item
    raise ValueError(f"Unknown temporal track: {raw_track}")


def _canonicalize_target_spec(
    adata,
    *,
    mode: str,
    raw_target_name: str,
    raw_gene_text: str,
    raw_signed_gene_text: str,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if mode == "gene":
        signed_items = _parse_optional_signed_gene_text(raw_signed_gene_text or raw_gene_text)
        if not signed_items:
            raise ValueError("Please provide at least one gene.")
        if len(signed_items) > GENE_LIMIT:
            raise ValueError(f"Gene perturbation supports at most {GENE_LIMIT} genes per job.")
        canonical_signed: list[str] = []
        missing_genes: list[str] = []
        display_tokens: list[str] = []
        seen: set[str] = set()
        for item in signed_items:
            gene, _sep, sign = item.partition(":")
            valid_genes, missing = _resolve_gene_list(adata, [gene])
            if missing:
                missing_genes.extend(missing)
                continue
            canonical = f"{valid_genes[0]}:{sign}"
            if canonical in seen:
                continue
            seen.add(canonical)
            canonical_signed.append(canonical)
            display_tokens.append(canonical)
        if missing_genes:
            warnings.append(f"Ignored missing genes: {', '.join(sorted(set(missing_genes)))}")
        if not canonical_signed:
            raise ValueError("None of the provided genes were found in the analysis dataset.")
        return {
            "signed_genes": canonical_signed,
            "internal_name": _custom_gene_set_internal_name(display_tokens),
        }, warnings

    target_name = str(raw_target_name or "").strip()
    if not target_name:
        raise ValueError("Please provide a target name.")

    if mode == "pathway":
        requested_genes = _split_gene_text(raw_gene_text)
        if not requested_genes:
            raise ValueError("Please provide at least one pathway gene.")
        valid_genes, missing_genes = _resolve_gene_list(adata, requested_genes)
        if missing_genes:
            warnings.append(f"Ignored missing genes: {', '.join(missing_genes)}")
        if not valid_genes:
            raise ValueError("None of the provided pathway genes were found in the analysis dataset.")
        return {"name": target_name, "genes": valid_genes}, warnings

    signed_items = _parse_signed_gene_text(raw_signed_gene_text)
    if not signed_items:
        raise ValueError("Please provide at least one signed drug gene.")
    canonical_signed: list[str] = []
    missing_genes: list[str] = []
    seen: set[str] = set()
    for item in signed_items:
        gene, _sep, sign = item.partition(":")
        valid_genes, missing = _resolve_gene_list(adata, [gene])
        if missing:
            missing_genes.extend(missing)
            continue
        canonical = f"{valid_genes[0]}:{sign}"
        if canonical in seen:
            continue
        seen.add(canonical)
        canonical_signed.append(canonical)
    if missing_genes:
        warnings.append(f"Ignored missing genes: {', '.join(sorted(set(missing_genes)))}")
    if not canonical_signed:
        raise ValueError("None of the provided drug-signature genes were found in the analysis dataset.")
    return {"name": target_name, "signed_genes": canonical_signed}, warnings


def _normalized_target_items(normalized_request: Mapping[str, Any]) -> dict[str, list[str]]:
    mode = str(normalized_request["mode"])
    target_spec = normalized_request["target_spec"]
    if mode == "gene":
        return {str(target_spec["internal_name"]): list(target_spec["signed_genes"])}
    if mode == "pathway":
        return {str(target_spec["name"]): list(target_spec["genes"])}
    return {str(target_spec["name"]): list(target_spec["signed_genes"])}


def _signature_sizes(normalized_request: Mapping[str, Any]) -> list[int]:
    items = _normalized_target_items(normalized_request)
    return [len(value) for value in items.values()]


def _resolved_execution_mode(
    requested_mode: str,
    *,
    analysis_n_obs: int,
    scope_cell_count: int,
    gpu_available: bool,
) -> tuple[str, str]:
    if requested_mode == "centroid":
        return "centroid", "gpu_centroid" if gpu_available else "cpu_centroid"
    if gpu_available:
        return "cell_exact", "gpu_exact"
    return "cell_exact", "cpu_exact_parallel"


def build_form_context(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    cache_key = _analysis_hash(analysis_path)
    with _FORM_CONTEXT_LOCK:
        cached = _FORM_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            _FORM_CONTEXT_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)
    prewarmed = _load_prewarmed_metadata_cache(analysis_path)
    if prewarmed is not None:
        form_context = dict(prewarmed["form_context"])
    else:
        adata, _dataset_path, _attribute_path = _read_analysis_adata(analysis_path)
        form_context = _build_form_context_from_adata(adata)
    with _FORM_CONTEXT_LOCK:
        _FORM_CONTEXT_CACHE[cache_key] = copy.deepcopy(form_context)
        _FORM_CONTEXT_CACHE.move_to_end(cache_key)
        while len(_FORM_CONTEXT_CACHE) > FORM_CONTEXT_CACHE_SIZE:
            _FORM_CONTEXT_CACHE.popitem(last=False)
    return form_context


def validate_custom_perturbation_payload(
    *,
    analysis_dir: str | Path,
    analysis_job_id: str,
    analysis_iteration: int,
    raw_payload: Mapping[str, Any],
    gpu_available: bool | None = None,
) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    adata, dataset_path, attribute_path = _read_analysis_adata(analysis_path)
    track_options = _extract_track_options(adata)
    mode = _normalize_mode(raw_payload.get("mode"))
    scope_kind = _normalize_scope_kind(raw_payload.get("scope_kind"))
    direction_mode = _normalize_direction_mode(raw_payload.get("direction_mode"))
    requested_execution_mode = _normalize_execution_mode(raw_payload.get("execution_mode"))
    perturb_change = float(raw_payload.get("perturb_change") or 0.5)
    if perturb_change == 1.0:
        raise ValueError("perturb_change must not be 1.0.")
    if gpu_available is None:
        gpu_available = _detect_gpu_available()

    selected_track = ""
    selected_track_label = ""
    selected_track_cluster = ""
    scope_cell_count = int(adata.n_obs)
    if scope_kind == "track":
        selected_track = str(raw_payload.get("track_key") or "").strip()
        if not selected_track:
            raise ValueError("Please select one temporal track.")
        track_entry = _resolve_track_option(track_options, selected_track)
        selected_track_label = track_entry["label"]
        selected_track_cluster = _track_cluster_id(selected_track)
        scope_cell_count = _track_scope_cell_count(adata, selected_track)

    target_spec, warnings = _canonicalize_target_spec(
        adata,
        mode=mode,
        raw_target_name=str(raw_payload.get("target_name") or ""),
        raw_gene_text=str(raw_payload.get("gene_text") or ""),
        raw_signed_gene_text=str(raw_payload.get("signed_gene_text") or ""),
    )
    execution_mode, resolved_execution_backend = _resolved_execution_mode(
        requested_execution_mode,
        analysis_n_obs=int(adata.n_obs),
        scope_cell_count=scope_cell_count,
        gpu_available=bool(gpu_available),
    )
    cpu_workers = _normalize_cpu_workers(raw_payload.get("cpu_workers"))
    queue_key = "gpu:0" if resolved_execution_backend.startswith("gpu_") else "cpu:0"
    normalized = {
        "analysis_job_id": str(analysis_job_id),
        "analysis_dir": str(analysis_path),
        "analysis_iteration": int(analysis_iteration),
        "dataset_path": str(dataset_path),
        "attribute_path": str(attribute_path) if attribute_path else None,
        "mode": mode,
        "scope_kind": scope_kind,
        "track_key": selected_track or None,
        "track_label": selected_track_label or None,
        "track_cluster_id": selected_track_cluster or None,
        "perturb_change": float(perturb_change),
        "direction_mode": direction_mode,
        "requested_execution_mode": requested_execution_mode,
        "execution_mode": execution_mode,
        "resolved_execution_backend": resolved_execution_backend,
        "cpu_workers": int(cpu_workers),
        "queue_key": queue_key,
        "target_spec": target_spec,
        "user_label": str(raw_payload.get("user_label") or "").strip() or None,
    }
    return {
        "ok": True,
        "normalized": normalized,
        "warnings": warnings,
        "estimated_scope_cell_count": scope_cell_count,
        "analysis_n_obs": int(adata.n_obs),
        "resolved_execution_mode": execution_mode,
        "resolved_execution_backend": resolved_execution_backend,
        "cpu_workers": int(cpu_workers),
        "queue_key": queue_key,
        "track_options": track_options,
    }


@dataclass
class PerturbationContext:
    skill_root: Path
    analysis_dir: Path
    adata: Any
    metadata: dict[str, Any]
    model_path: Path
    training_params_path: Path
    cache_root: Path
    analysis_iteration: int

    def __post_init__(self) -> None:
        self._runner_cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._source_random_background = copy.deepcopy(self.adata.uns.get("random_background_perturbation_deltaD", {}))
        self._source_drug_overlap_genes = copy.deepcopy(self.adata.uns.get("data_drug_overlap_genes", {}))
        self._source_pathway_overlap_genes = copy.deepcopy(self.adata.uns.get("data_pathway_overlap_genes", {}))
        self._analysis_direction_metadata = _extract_analysis_direction_metadata(self.adata)

    def get_runner(self, execution_backend: str):
        with self._lock:
            runner_mode = "exact" if execution_backend in {"gpu_exact", "cpu_exact_parallel", "cell_exact"} else "centroid"
            runner = self._runner_cache.get(runner_mode)
            if runner is not None:
                return runner
            if runner_mode == "exact":
                device_override = "cpu" if execution_backend == "cpu_exact_parallel" else (
                    "cuda:0" if execution_backend == "gpu_exact" else None
                )
                try:
                    runner = ExactPerturbation(
                        self.adata,
                        self.model_path,
                        self.analysis_dir / "idrem",
                        config_path=self.training_params_path,
                        device_override=device_override,
                    )
                except TypeError as exc:
                    if 'device_override' not in str(exc):
                        raise
                    runner = ExactPerturbation(
                        self.adata,
                        self.model_path,
                        self.analysis_dir / "idrem",
                        config_path=self.training_params_path,
                    )
            else:
                runner = CentroidPerturbation(
                    self.adata,
                    self.model_path,
                    self.analysis_dir / "idrem",
                )
            self._runner_cache[runner_mode] = runner
            return runner

    def _runner_clear_keys(self, runner, *, backend_mode: str) -> None:
        keys = ["random_background_perturbation_deltaD"]
        if backend_mode == "single_gene":
            keys.extend(
                [
                    "single_gene_perturbation_deltaD",
                    "single_gene_perturbation_score",
                ]
            )
        elif backend_mode == "pathway":
            keys.extend(
                [
                    "pathway_perturbation_deltaD",
                    "pathway_perturbation_score",
                    "data_pathway_overlap_genes",
                    "pathway_gene_len_dict",
                ]
            )
        else:
            keys.extend(
                [
                    "drug_perturbation_deltaD",
                    "drug_perturbation_score",
                    "data_drug_overlap_genes",
                    "drug-gene_len_dict",
                ]
            )
        for key in keys:
            runner.adata.uns.pop(key, None)

    def _set_backend_items(self, runner, *, mode: str, items: dict[str, list[str]]) -> None:
        if mode == "single_gene":
            return
        if mode == "pathway":
            runner.adata.uns["data_pathway_overlap_genes"] = copy.deepcopy(items)
            runner.adata.uns["pathway_gene_len_dict"] = {key: len(value) for key, value in items.items()}
        else:
            runner.adata.uns["data_drug_overlap_genes"] = copy.deepcopy(items)
            runner.adata.uns["drug-gene_len_dict"] = {key: len(value) for key, value in items.items()}

    def _background_cache_path(
        self,
        *,
        backend_mode: str,
        execution_mode: str,
        direction_mode: str,
        perturb_change: float,
        sizes: Sequence[int],
    ) -> Path:
        signature = ",".join(str(size) for size in sorted(int(size) for size in sizes))
        filename = (
            f"{backend_mode}_{execution_mode}_{direction_mode}_pc{perturb_change:g}_{signature}"
            f"_rt{DEFAULT_BACKGROUND_RANDOM_TIMES}_v{BACKGROUND_CACHE_VERSION}.pkl"
        )
        return self.cache_root / "backgrounds" / filename

    def _reuse_precomputed_background(
        self,
        *,
        backend_mode: str,
        direction_mode: str,
        perturb_change: float,
        sizes: Sequence[int],
        count_per_size: int,
    ) -> tuple[dict[str, Any], dict[int, int]]:
        if backend_mode == "pathway":
            source_items = self._source_pathway_overlap_genes
        elif backend_mode == "drug":
            source_items = self._source_drug_overlap_genes
        else:
            return {}, {}
        source_background = self._source_random_background
        if not isinstance(source_items, dict) or not isinstance(source_background, dict):
            return {}, {}

        expected_forward_key = str(perturb_change)
        expected_reverse_key = str(1 / float(perturb_change))
        if expected_forward_key not in source_background:
            return {}, {size: 0 for size in sorted({max(1, int(size)) for size in sizes})}
        if direction_mode == "two_sided" and expected_reverse_key not in source_background:
            return {}, {size: 0 for size in sorted({max(1, int(size)) for size in sizes})}

        source_background_change = self._analysis_direction_metadata.get("background_perturb_change")
        if source_background_change not in (None, ""):
            try:
                if not np.isclose(float(source_background_change), float(perturb_change)):
                    return {}, {size: 0 for size in sorted({max(1, int(size)) for size in sizes})}
            except Exception:
                return {}, {size: 0 for size in sorted({max(1, int(size)) for size in sizes})}

        source_direction_mode = str(self._analysis_direction_metadata.get("background_direction_mode") or "").strip()
        if source_direction_mode:
            source_direction_mode = _normalize_direction_mode(source_direction_mode)
        else:
            try:
                source_direction_mode = infer_direction_mode(source_background, float(perturb_change))
            except Exception:
                source_direction_mode = ""
        if source_direction_mode and source_direction_mode != direction_mode:
            return {}, {size: 0 for size in sorted({max(1, int(size)) for size in sizes})}

        unique_sizes = sorted({max(1, int(size)) for size in sizes})
        size_to_keys: dict[int, list[str]] = {size: [] for size in unique_sizes}
        for key, values in source_items.items():
            if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                item_size = len(values)
            else:
                continue
            if item_size in size_to_keys:
                size_to_keys[item_size].append(str(key))

        selected_keys: list[str] = []
        reused_counts: dict[int, int] = {}
        for size in unique_sizes:
            keys = size_to_keys.get(size, [])
            take = min(len(keys), int(count_per_size))
            reused_counts[size] = take
            selected_keys.extend(keys[:take])
        if not selected_keys:
            return {}, reused_counts

        payload: dict[str, Any] = {}
        for direction_key, track_payload in source_background.items():
            if not isinstance(track_payload, dict):
                continue
            payload[str(direction_key)] = {}
            for track_key, item_payload in track_payload.items():
                if not isinstance(item_payload, dict):
                    continue
                filtered = {
                    str(item_key): copy.deepcopy(item_payload[item_key])
                    for item_key in selected_keys
                    if item_key in item_payload
                }
                if filtered:
                    payload[str(direction_key)][str(track_key)] = filtered
        return payload, reused_counts

    def _load_or_build_background(
        self,
        *,
        runner,
        backend_mode: str,
        execution_mode: str,
        direction_mode: str,
        perturb_change: float,
        sizes: Sequence[int],
        use_cuda: bool,
        device: str,
        log: Callable[[str], None],
    ) -> dict[str, Any]:
        cache_path = self._background_cache_path(
            backend_mode=backend_mode,
            execution_mode=execution_mode,
            direction_mode=direction_mode,
            perturb_change=perturb_change,
            sizes=sizes,
        )
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                return pickle.load(handle)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        unique_sizes = sorted({max(1, int(size)) for size in sizes})
        reused_payload, reused_counts = self._reuse_precomputed_background(
            backend_mode=backend_mode,
            direction_mode=direction_mode,
            perturb_change=perturb_change,
            sizes=unique_sizes,
            count_per_size=DEFAULT_BACKGROUND_RANDOM_TIMES,
        )
        missing_counts = {
            size: max(0, DEFAULT_BACKGROUND_RANDOM_TIMES - int(reused_counts.get(size, 0)))
            for size in unique_sizes
        }
        if all(count == 0 for count in missing_counts.values()):
            log(
                f"[UNAGI-CUSTOM-PERTURB] stage=background status=DONE backend={backend_mode} "
                f"source=precomputed reuse_only=1 sizes={','.join(str(size) for size in unique_sizes)}"
            )
            with cache_path.open("wb") as handle:
                pickle.dump(reused_payload, handle)
            return reused_payload

        background_items: dict[str, list[str]] = {}
        genes = [str(name) for name in runner.adata.var_names.astype(str).tolist()]
        for size in unique_sizes:
            needed = int(missing_counts.get(size, 0))
            if needed <= 0:
                continue
            selected = genes[:size]
            start_index = int(reused_counts.get(size, 0))
            for index in range(start_index, start_index + needed):
                key = f"bg_size{size}_{index}"
                if backend_mode == "pathway":
                    background_items[key] = selected
                else:
                    background_items[key] = [f"{gene}:+" for gene in selected]

        generated_payload: dict[str, Any] = {}
        if background_items:
            self._set_backend_items(runner, mode=backend_mode, items=background_items)
            log(
                f"[UNAGI-CUSTOM-PERTURB] stage=background status=START backend={backend_mode} "
                f"count={len(background_items)} reused={sum(reused_counts.values())} "
                f"sizes={','.join(str(size) for size in unique_sizes)}"
            )
            if backend_mode == "pathway":
                runner.run(
                    "random_pathway_background",
                    0.5,
                    inplace=True,
                    CUDA=use_cuda,
                    device=device,
                    direction_mode=direction_mode,
                )
            else:
                runner.run(
                    "random_drug_background",
                    0.5,
                    inplace=True,
                    random_times=max(len(background_items), 1),
                    random_genes=None,
                    CUDA=use_cuda,
                    device=device,
                    direction_mode=direction_mode,
                )
            log("[UNAGI-CUSTOM-PERTURB] stage=background status=DONE")
            generated_payload = copy.deepcopy(runner.adata.uns.get("random_background_perturbation_deltaD", {}))
        payload = merge_directional_payloads(reused_payload, generated_payload)
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return payload

    def _copy_single_gene_background(self, runner, *, perturb_change: float, direction_mode: str) -> dict[str, Any]:
        key = str(perturb_change)
        single_gene_payload = runner.adata.uns.get("single_gene_perturbation_deltaD", {})
        if key not in single_gene_payload:
            raise ValueError(f"Missing single_gene perturbation delta payload for {key}")
        payload = {key: copy.deepcopy(single_gene_payload[key])}
        if direction_mode == "two_sided":
            reverse_key = str(1 / float(perturb_change))
            if reverse_key in single_gene_payload:
                payload[reverse_key] = copy.deepcopy(single_gene_payload[reverse_key])
        runner.adata.uns["random_background_perturbation_deltaD"] = copy.deepcopy(payload)
        return payload

    def _extract_result_table(
        self,
        runner,
        *,
        request_payload: Mapping[str, Any],
        backend_mode: str,
    ) -> tuple[pd.DataFrame, str]:
        score_key = f"{backend_mode}_perturbation_score"
        perturb_key = str(request_payload["perturb_change"])
        score_bucket = runner.adata.uns.get(score_key, {}).get(perturb_key, {})
        if not isinstance(score_bucket, dict) or not score_bucket:
            raise ValueError(f"No perturbation score bucket found for {score_key}[{perturb_key}]")

        selected_track = "overall"
        if request_payload["scope_kind"] == "track":
            raw_track = str(request_payload.get("track_key") or "")
            cluster_id = str(request_payload.get("track_cluster_id") or "")
            if raw_track in score_bucket:
                selected_track = raw_track
            else:
                for key in score_bucket:
                    if str(key).split("-")[0] == cluster_id:
                        selected_track = str(key)
                        break
                else:
                    selected_track = next(iter(score_bucket.keys()))
        elif "overall" not in score_bucket:
            selected_track = next(iter(score_bucket.keys()))

        from UNAGI.plotting.manuscript_figures import extract_perturbation_table

        table = extract_perturbation_table(
            runner.adata,
            mode=backend_mode,
            perturb_change=request_payload["perturb_change"],
            track=selected_track,
        )
        if request_payload["mode"] == "gene":
            label_column = "pathways" if "pathways" in table.columns else ("compound" if "compound" in table.columns else None)
            internal_name = str(request_payload["target_spec"].get("internal_name") or "").strip()
            display_name = ", ".join(str(item) for item in request_payload["target_spec"].get("signed_genes") or [])
            if label_column:
                table = table.copy()
                if internal_name:
                    table = table[table[label_column].astype(str) == internal_name]
                table = table.rename(columns={label_column: "gene set"})
                if display_name:
                    table["gene set"] = display_name
        elif request_payload["mode"] == "pathway" and "compound" in table.columns:
            table = table.rename(columns={"compound": "pathways"})
        return table, selected_track

    def run_request(
        self,
        request_payload: Mapping[str, Any],
        *,
        result_dir: Path,
        log: Callable[[str], None],
    ) -> dict[str, Any]:
        backend_mode = BACKEND_MODE_BY_MODE[str(request_payload["mode"])]
        execution_mode = str(request_payload["execution_mode"])
        resolved_execution_backend = str(request_payload.get("resolved_execution_backend") or execution_mode)
        runner = self.get_runner(resolved_execution_backend)
        self._runner_clear_keys(runner, backend_mode=backend_mode)
        use_cuda = resolved_execution_backend.startswith("gpu_")
        device = "cuda:0" if use_cuda else "cpu"
        cpu_workers = int(request_payload.get("cpu_workers") or default_cpu_workers())

        log("[UNAGI-CUSTOM-PERTURB] stage=load_context status=START")
        log(
            f"[UNAGI-CUSTOM-PERTURB] stage=load_context status=DONE "
            f"execution_mode={execution_mode} resolved_backend={resolved_execution_backend} "
            f"device={device} cpu_workers={cpu_workers}"
        )
        log("[UNAGI-CUSTOM-PERTURB] stage=resolve_targets status=START")
        items = _normalized_target_items(request_payload)
        self._set_backend_items(runner, mode=backend_mode, items=items)
        log(
            f"[UNAGI-CUSTOM-PERTURB] stage=resolve_targets status=DONE mode={request_payload['mode']} "
            f"targets={len(items)}"
        )

        background = self._load_or_build_background(
            runner=runner,
            backend_mode=backend_mode,
            execution_mode=resolved_execution_backend,
            direction_mode=str(request_payload["direction_mode"]),
            perturb_change=float(request_payload["perturb_change"]),
            sizes=_signature_sizes(request_payload),
            use_cuda=use_cuda,
            device=device,
            log=log,
        )
        runner.adata.uns["random_background_perturbation_deltaD"] = copy.deepcopy(background)
        self._set_backend_items(runner, mode=backend_mode, items=items)

        log(f"[UNAGI-CUSTOM-PERTURB] stage=run status=START total={len(items)}")
        run_mode = backend_mode
        if resolved_execution_backend == "cpu_exact_parallel":
            delta_payload = run_exact_parallel_delta(
                dataset_path=Path(str(request_payload["dataset_path"])).expanduser().resolve(),
                attribute_path=Path(str(request_payload["attribute_path"])).expanduser().resolve()
                if request_payload.get("attribute_path")
                else None,
                model_path=self.model_path,
                training_params_path=self.training_params_path,
                idrem_dir=self.analysis_dir / "idrem",
                backend_mode=backend_mode,
                items=items,
                perturb_change=float(request_payload["perturb_change"]),
                direction_mode=str(request_payload["direction_mode"]),
                cpu_workers=cpu_workers,
            )
            runner.adata.uns[f"{backend_mode}_perturbation_deltaD"] = copy.deepcopy(delta_payload)
        else:
            runner.run(
                run_mode,
                float(request_payload["perturb_change"]),
                inplace=True,
                CUDA=use_cuda,
                device=device,
                direction_mode=str(request_payload["direction_mode"]),
            )
        batch_meta = getattr(runner, "last_gpu_batch_meta", None) or {}
        log(
            f"[UNAGI-CUSTOM-PERTURB] stage=run finished={len(items)}/{len(items)} total={len(items)} "
            f"backend={resolved_execution_backend} batch_size={batch_meta.get('batch_size', 1)} "
            f"batched={str(batch_meta.get('batched', False)).lower()}"
        )
        log(f"[UNAGI-CUSTOM-PERTURB] stage=run status=DONE total={len(items)}")

        perturbed_tracks: Any = "all"
        overall = True
        if request_payload["scope_kind"] == "track":
            perturbed_tracks = [str(request_payload["track_cluster_id"])]
            overall = False

        log("[UNAGI-CUSTOM-PERTURB] stage=score status=START")
        runner.analysis(
            run_mode,
            float(request_payload["perturb_change"]),
            perturbed_tracks=perturbed_tracks,
            overall_perturbation_analysis=overall,
            direction_mode=str(request_payload["direction_mode"]),
        )
        log("[UNAGI-CUSTOM-PERTURB] stage=score status=DONE")
        table, resolved_track_key = self._extract_result_table(
            runner,
            request_payload=request_payload,
            backend_mode=backend_mode,
        )
        if "pval_adjusted" not in table.columns and "pval" in table.columns:
            table = table.copy()
            table["pval_adjusted"] = _bh_adjust(pd.to_numeric(table["pval"], errors="coerce").fillna(1.0).tolist())
        if "pval_adjusted" not in table.columns or not pd.to_numeric(table["pval_adjusted"], errors="coerce").notna().any():
            raise ValueError(
                "The custom perturbation did not produce valid p-values against the random background."
            )
        if table.empty:
            raise ValueError("The custom perturbation produced an empty result table.")

        log("[UNAGI-CUSTOM-PERTURB] stage=save status=START")
        metadata = {
            "resolved_track_key": resolved_track_key,
            "track_label": request_payload.get("track_label") or "Overall",
            "analysis_iteration": self.analysis_iteration,
            "execution_mode": execution_mode,
            "resolved_execution_backend": resolved_execution_backend,
            "cpu_workers": int(cpu_workers),
            "direction_mode": request_payload["direction_mode"],
            "scope_kind": request_payload["scope_kind"],
        }
        summary = _write_result_bundle(
            result_dir=result_dir,
            normalized_request=dict(request_payload),
            table=table,
            metadata=metadata,
        )
        gc.collect()
        log("[UNAGI-CUSTOM-PERTURB] stage=save status=DONE")
        return summary


class PerturbationContextCache:
    def __init__(self, *, skill_root: Path, max_contexts: int = 1) -> None:
        self.skill_root = Path(skill_root).expanduser().resolve()
        self.max_contexts = max(1, int(max_contexts))
        self._lock = threading.Lock()
        self._contexts: "OrderedDict[tuple[Any, ...], PerturbationContext]" = OrderedDict()

    def _build_context(self, request_payload: Mapping[str, Any]) -> PerturbationContext:
        analysis_dir = Path(str(request_payload["analysis_dir"])).expanduser().resolve()
        adata, _dataset_path, _attribute_path = _read_analysis_adata(analysis_dir)
        training_params_path = analysis_dir / "model_save" / "training_parameters.json"
        if not training_params_path.exists():
            raise FileNotFoundError(f"training_parameters.json not found: {training_params_path}")
        model_path = _pick_model_path(analysis_dir / "model_save", int(request_payload["analysis_iteration"]))
        cache_root = self.skill_root / "outputs" / "perturbation_cache" / _analysis_hash(analysis_dir)
        return PerturbationContext(
            skill_root=self.skill_root,
            analysis_dir=analysis_dir,
            adata=adata,
            metadata=_extract_metadata(adata),
            model_path=model_path,
            training_params_path=training_params_path,
            cache_root=cache_root,
            analysis_iteration=int(request_payload["analysis_iteration"]),
        )

    def get(self, request_payload: Mapping[str, Any]) -> PerturbationContext:
        key = _analysis_identity(Path(str(request_payload["analysis_dir"])))
        with self._lock:
            cached = self._contexts.get(key)
            if cached is not None:
                self._contexts.move_to_end(key)
                return cached
        context = self._build_context(request_payload)
        with self._lock:
            self._contexts[key] = context
            self._contexts.move_to_end(key)
            while len(self._contexts) > self.max_contexts:
                self._contexts.popitem(last=False)
        return context


def execute_custom_perturbation_job(
    *,
    skill_root: Path,
    request_payload: Mapping[str, Any],
    result_dir: Path,
    log_handle,
    context_cache: PerturbationContextCache | None = None,
) -> dict[str, Any]:
    cache = context_cache or PerturbationContextCache(skill_root=skill_root)
    context = cache.get(request_payload)

    def log(message: str) -> None:
        log_handle.write(f"{message}\n")
        log_handle.flush()

    with redirect_stdout(log_handle), redirect_stderr(log_handle):
        return context.run_request(request_payload, result_dir=result_dir, log=log)
