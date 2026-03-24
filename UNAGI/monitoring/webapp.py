from __future__ import annotations

import copy
import importlib
import json
import mimetypes
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, urlencode

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query, Request, status as http_status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from UNAGI.workflow import PreflightError, run_workflow_from_payload

from .figure_export import STATIC_PDF_EXPORTS, export_static_publication_pdf
from .interactive_analysis import AnalysisExplorerService
from .share_bundle import export_analysis_share_bundle
from .service import ACTIVE_STATUSES, JobMonitorService
from .visualization_pipeline import generate_analysis_visualizations

TEXT_PREVIEW_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".html",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".text",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

TRAINING_GUIDED_FIELDS = [
    "data.path",
    "data.stage_key",
    "data.total_stage",
    "data.label_key",
    "random_seed",
    "training.task",
    "training.dist",
    "training.GPU",
    "training.device",
    "training.epoch_initial",
    "training.epoch_iter",
    "training.max_iter",
    "training.BATCHSIZE",
    "training.adversarial",
    "training.GCN",
    "species",
    "run.CPO",
    "run.connect_edges_cutoff",
]

ANALYSIS_GUIDED_FIELDS = [
    "random_background_sampling_times",
    "run_perturbation",
    "perturb_change",
    "direction_mode",
    "overall_perturbation_analysis",
    "perturbed_tracks",
    "customized_pathway",
    "customized_drug",
    "cmap_dir",
    "ignore_pathway_perturabtion",
    "ignore_drug_perturabtion",
    "centroid",
    "ignore_hcmarkers",
    "ignore_dynamic_markers",
]

PATH_FIELD_SPECS = {
    "data.path": "training_data",
    "customized_pathway": "path",
    "customized_drug": "path",
    "cmap_dir": "dir",
    "template_path": "path",
}

VISUALIZATION_GROUP_SPECS = [
    {
        "name": "temporal_dynamics_overview",
        "label": "Temporal dynamics overview",
        "members": ["stage_umap_grid", "stage_composition", "temporal_dynamics_graph"],
    },
    {
        "name": "dynamic_markers",
        "label": "Dynamic markers",
        "members": ["dynamic_marker_table", "dynamic_marker_trajectories", "dynamic_marker_heatmap"],
    },
    {
        "name": "hierarchical_markers",
        "label": "Hierarchical markers",
        "members": ["hierarchical_marker_heatmap", "hc_dendrograms"],
    },
]

TEMPORAL_OVERVIEW_ENTRY_ORDER = [
    "stage_umap_grid",
    "temporal_dynamics_graph",
    "stage_composition",
]

VISUALIZATION_ENTRY_LABELS = {
    "stage_umap_grid": "Stage UMAP",
    "stage_composition": "Stage composition",
    "temporal_dynamics_graph": "Temporal dynamics graph",
    "dynamic_marker_table": "Dynamic marker table",
    "dynamic_marker_trajectories": "Dynamic marker trajectories",
    "dynamic_marker_heatmap": "Dynamic marker heatmap",
    "hierarchical_marker_heatmap": "Hierarchical marker heatmap",
    "hc_dendrograms": "Hierarchical dendrograms",
    "pathway_ranked_bar": "Pathway perturbation",
    "drug_ranked_bar": "Drug perturbation",
    "single_gene_ranked_bar": "Single-gene perturbation",
    "gene_combo_ranked_bar": "Gene-combo perturbation",
}


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _safe_path(path_value: str, roots: Iterable[Path]) -> Path:
    path = Path(path_value).expanduser().resolve()
    for root in roots:
        root = root.resolve()
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    raise HTTPException(status_code=404, detail="Path is outside the allowed roots")


def _safe_user_path(path_value: str, roots: Iterable[Path], repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return _safe_path(str(path), roots)


def _summary_counts(jobs: list[dict]) -> list[dict]:
    active = sum(1 for job in jobs if job["status"] in ACTIVE_STATUSES)
    succeeded = sum(1 for job in jobs if job["status"] == "succeeded")
    failed = sum(1 for job in jobs if job["status"] == "failed")
    cancelled = sum(1 for job in jobs if job["status"] == "cancelled")
    return [
        {"label": "Active Jobs", "value": active},
        {"label": "Succeeded", "value": succeeded},
        {"label": "Failed", "value": failed},
        {"label": "Cancelled", "value": cancelled},
    ]


def _build_dashboard_view(all_jobs: list[dict], matched_jobs: list[dict]) -> dict[str, Any]:
    matched_ids = {job["job_id"] for job in matched_jobs}
    job_map = {job["job_id"]: job for job in all_jobs}
    tasks = [job for job in all_jobs if job["job_kind"] == "train"]
    analyses = [job for job in all_jobs if job["job_kind"] == "analysis"]

    analyses_by_parent: dict[str, list[dict]] = {}
    orphan_analyses: list[dict] = []
    for analysis in analyses:
        parent_id = analysis.get("parent_job_id")
        parent = job_map.get(parent_id)
        if parent and parent.get("job_kind") == "train":
            analyses_by_parent.setdefault(str(parent_id), []).append(analysis)
        else:
            orphan_analyses.append(analysis)

    task_entries: list[dict[str, Any]] = []
    for task in tasks:
        child_analyses = analyses_by_parent.get(task["job_id"], [])
        visible_analyses = [analysis for analysis in child_analyses if analysis["job_id"] in matched_ids]
        include_task = task["job_id"] in matched_ids or bool(visible_analyses)
        if not include_task:
            continue
        task_entries.append(
            {
                "task": task,
                "analyses": visible_analyses,
                "total_analyses": len(child_analyses),
                "analysis_open": task.get("status") in ACTIVE_STATUSES
                or any(analysis.get("status") in ACTIVE_STATUSES for analysis in child_analyses),
            }
        )

    visible_orphans = [analysis for analysis in orphan_analyses if analysis["job_id"] in matched_ids]

    return {
        "tasks": task_entries,
        "orphan_analyses": visible_orphans,
        "task_count": len(task_entries),
        "analysis_count": sum(len(item["analyses"]) for item in task_entries) + len(visible_orphans),
    }


def _is_text_previewable(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("text/"):
        return True
    return path.suffix.lower() in TEXT_PREVIEW_SUFFIXES


def _load_job_backend(skill_dir: Path):
    scripts_dir = skill_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("unagi_job")


def _load_custom_perturbation_backend():
    return importlib.import_module("UNAGI.monitoring.custom_perturbation")


def _nested_get(payload: dict[str, Any], dotted_key: str, default: Any = "") -> Any:
    node: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _yaml_text(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def _stringify_form_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _base_training_initial(payload: dict[str, Any], *, template_path: str) -> dict[str, str]:
    initial = {"template_path": template_path}
    for key in TRAINING_GUIDED_FIELDS:
        initial[key] = _stringify_form_value(_nested_get(payload, key))
    return initial


def _base_analysis_initial(payload: dict[str, Any], *, template_path: str, iteration: int) -> dict[str, str]:
    initial = {"template_path": template_path, "iteration": str(iteration)}
    for key in ANALYSIS_GUIDED_FIELDS:
        initial[key] = _stringify_form_value(payload.get(key))
    return initial


def _parse_yaml_text(raw_text: str) -> dict[str, Any]:
    if not raw_text.strip():
        return {}
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Advanced YAML must be a mapping")
    return parsed


class ParsedForm(dict):
    def getlist(self, key: str) -> list[str]:
        value = self.get(key)
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]


async def _parse_request_form(request: Request) -> ParsedForm:
    body = await request.body()
    if not body:
        return ParsedForm()
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    data = ParsedForm()
    for key, values in parsed.items():
        if len(values) == 1:
            data[key] = values[0]
        else:
            data[key] = values
    return data


def _apply_overrides(job_backend, payload: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
    raw = [
        f"{key}={value}"
        for key, value in overrides.items()
        if value not in ("", None)
    ]
    if not raw:
        return payload
    return job_backend.apply_set_overrides(copy.deepcopy(payload), raw)


def _selection_kind_matches(path: Path, kind: str) -> bool:
    if kind == "training_data":
        if path.is_file():
            return path.suffix.lower() == ".h5ad"
        if path.is_dir():
            return any(child.suffix.lower() == ".h5ad" for child in path.iterdir())
        return False
    if kind == "dir":
        return path.is_dir()
    if kind == "file":
        return path.is_file()
    return path.exists()


def _resolve_path_field(
    field_name: str,
    raw_value: str,
    *,
    repo_root: Path,
    allowed_roots: list[Path],
) -> str:
    if raw_value in ("", None):
        return ""
    resolved = _safe_user_path(raw_value, allowed_roots, repo_root)
    kind = PATH_FIELD_SPECS.get(field_name)
    if kind and not _selection_kind_matches(resolved, kind):
        raise ValueError(f"{field_name} must be a valid {kind.replace('_', ' ')} path")
    return str(resolved)


def _training_payload_from_form(
    job_backend,
    form_data,
    *,
    repo_root: Path,
    allowed_roots: list[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_template_path = str(form_data.get("template_path") or "").strip()
    template_path = (
        _resolve_path_field("template_path", raw_template_path, repo_root=repo_root, allowed_roots=allowed_roots)
        if raw_template_path
        else ""
    )
    advanced_yaml = form_data.get("advanced_yaml") or ""
    job_name = (form_data.get("job_name") or "").strip() or None

    if advanced_yaml.strip():
        payload = _parse_yaml_text(advanced_yaml)
    else:
        payload, _ = job_backend.load_training_template(repo_root, template_path or None)

    overrides: dict[str, str] = {}
    for field in TRAINING_GUIDED_FIELDS:
        raw_value = form_data.get(field)
        if raw_value in (None, ""):
            continue
        if field in PATH_FIELD_SPECS:
            overrides[field] = _resolve_path_field(
                field,
                str(raw_value),
                repo_root=repo_root,
                allowed_roots=allowed_roots,
            )
        else:
            overrides[field] = str(raw_value)

    payload = _apply_overrides(job_backend, payload, overrides)
    if "data" not in payload:
        payload["data"] = {}
    if "data.path" not in overrides and _nested_get(payload, "data.path"):
        payload["data"]["path"] = _resolve_path_field(
            "data.path",
            str(payload["data"]["path"]),
            repo_root=repo_root,
            allowed_roots=allowed_roots,
        )

    meta = {
        "job_name": job_name,
        "template_path": template_path,
        "advanced_yaml": advanced_yaml,
    }
    return payload, meta


def _analysis_payload_from_form(
    job_backend,
    form_data,
    *,
    repo_root: Path,
    allowed_roots: list[Path],
    iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_template_path = str(form_data.get("template_path") or "").strip()
    template_path = (
        _resolve_path_field("template_path", raw_template_path, repo_root=repo_root, allowed_roots=allowed_roots)
        if raw_template_path
        else ""
    )
    advanced_yaml = form_data.get("advanced_yaml") or ""
    job_name = (form_data.get("job_name") or "").strip() or None

    if advanced_yaml.strip():
        payload = _parse_yaml_text(advanced_yaml)
    else:
        payload, _ = job_backend._load_analysis_template(repo_root, template_path or None)

    overrides: dict[str, str] = {}
    for field in ANALYSIS_GUIDED_FIELDS:
        raw_value = form_data.get(field)
        if raw_value in (None, ""):
            continue
        if field in PATH_FIELD_SPECS:
            overrides[field] = _resolve_path_field(
                field,
                str(raw_value),
                repo_root=repo_root,
                allowed_roots=allowed_roots,
            )
        else:
            overrides[field] = str(raw_value)

    payload = _apply_overrides(job_backend, payload, overrides)
    payload["iteration"] = int(iteration)
    meta = {
        "job_name": job_name,
        "template_path": template_path,
        "advanced_yaml": advanced_yaml,
    }
    return payload, meta


def _custom_perturbation_payload_from_form(form_data) -> dict[str, Any]:
    return {
        "mode": str(form_data.get("mode") or "gene"),
        "scope_kind": str(form_data.get("scope_kind") or "overall"),
        "track_key": str(form_data.get("track_key") or "").strip() or None,
        "perturb_change": str(form_data.get("perturb_change") or "0.5"),
        "direction_mode": str(form_data.get("direction_mode") or "two_sided"),
        "execution_mode": str(form_data.get("execution_mode") or "auto"),
        "target_name": str(form_data.get("target_name") or "").strip(),
        "gene_text": str(form_data.get("gene_text") or ""),
        "signed_gene_text": str(form_data.get("signed_gene_text") or ""),
    }


def _custom_perturbation_jobs(job: dict, *, job_backend=None, repo_root: Path | None = None) -> list[dict]:
    jobs = [
        dict(child)
        for child in job.get("child_jobs") or []
        if str(child.get("job_kind")) == "custom_perturbation"
    ]
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    if job_backend is not None and repo_root is not None:
        for child in jobs:
            diagnostics = None
            try:
                diagnostics = job_backend.custom_perturbation_queue_diagnostics(
                    repo_root=repo_root,
                    job_file_path=child["job_file_path"],
                )
            except Exception:
                diagnostics = None
            child["queue_diagnostics"] = diagnostics
            child["display_status"] = (
                str(diagnostics.get("derived_status"))
                if diagnostics and str(child.get("status") or "") == "queued"
                else str(child.get("status") or "")
            )
    return jobs


def _stale_custom_perturbation_error(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    request = result.get("request") or {}
    if str(request.get("mode") or "") != "gene":
        return None
    target_spec = request.get("target_spec") or {}
    requested_genes = set()
    for item in (target_spec.get("genes") or []):
        token = str(item).strip()
        if token:
            requested_genes.add(token.upper())
    for item in (target_spec.get("signed_genes") or []):
        token = str(item).partition(":")[0].strip()
        if token:
            requested_genes.add(token.upper())
    if not requested_genes:
        return None
    preview = result.get("preview") or {}
    columns = [str(column) for column in (preview.get("columns") or [])]
    rows = list(preview.get("rows") or [])
    if not columns or not rows or "gene" not in columns:
        return None
    gene_index = columns.index("gene")
    preview_genes = {
        str(row[gene_index]).strip().upper()
        for row in rows
        if isinstance(row, list) and len(row) > gene_index and str(row[gene_index]).strip()
    }
    if not preview_genes:
        return None
    if all(name.startswith("BG_") for name in preview_genes):
        return "This gene-mode custom perturbation result was created before the backend fix and needs to be rerun."
    if preview_genes.isdisjoint(requested_genes):
        return "This gene-mode custom perturbation result does not match the requested genes and needs to be rerun."
    return None


def _selected_custom_perturbation_job_id(source: Mapping[str, Any] | None) -> str | None:
    if source is None:
        return None
    selected_job_id = str(source.get("selected_custom_job_id") or "").strip()
    return selected_job_id or None


def _custom_perturbation_detail_context(
    *,
    service,
    custom_perturbation_backend,
    job_backend,
    repo_root: Path,
    analysis_job: dict,
    selected_job_id: str | None,
) -> dict[str, Any]:
    custom_jobs = _custom_perturbation_jobs(analysis_job, job_backend=job_backend, repo_root=repo_root)
    if not custom_jobs:
        return {
            "selected_custom_perturbation_job": None,
            "selected_custom_perturbation_result": None,
            "selected_custom_perturbation_error": None,
            "selected_custom_perturbation_preview": None,
        }
    if selected_job_id:
        selected_job_summary = next((item for item in custom_jobs if str(item.get("job_id")) == selected_job_id), None)
        if selected_job_summary is None:
            selected_job = None
        else:
            try:
                selected_job = service.get_job(selected_job_id)
            except FileNotFoundError:
                selected_job = None
    else:
        selected_job = None
    if selected_job is None:
        return {
            "selected_custom_perturbation_job": None,
            "selected_custom_perturbation_result": None,
            "selected_custom_perturbation_error": None,
            "selected_custom_perturbation_preview": None,
        }

    result = None
    result_error = None
    preview = None
    result_path = Path(str(selected_job.get("output_path") or selected_job.get("analysis_output_dir") or "")).expanduser()
    if result_path.exists():
        try:
            result = custom_perturbation_backend.load_result_bundle(result_path)
            stale_error = _stale_custom_perturbation_error(result)
            if stale_error:
                result_error = stale_error
                result = None
            else:
                preview = _custom_perturbation_score_preview(result)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            result_error = str(exc)
    selected_summary = next((item for item in custom_jobs if str(item.get("job_id")) == str(selected_job.get("job_id"))), None)
    if selected_summary and selected_summary.get("queue_diagnostics") is not None:
        selected_job["queue_diagnostics"] = selected_summary["queue_diagnostics"]
        selected_job["display_status"] = selected_summary.get("display_status", selected_job.get("status"))
    return {
        "selected_custom_perturbation_job": selected_job,
        "selected_custom_perturbation_result": result,
        "selected_custom_perturbation_error": result_error,
        "selected_custom_perturbation_preview": preview,
    }


def _custom_perturbation_score_preview(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    preview = result.get("preview") or {}
    columns = list(preview.get("columns") or [])
    rows = list(preview.get("rows") or [])
    if not columns or not rows:
        return None
    item_column = next(
        (name for name in ("gene set", "gene", "pathways", "compound", "item") if name in columns),
        None,
    )
    score_column = "perturbation score" if "perturbation score" in columns else None
    pvalue_column = "pval_adjusted" if "pval_adjusted" in columns else ("pval" if "pval" in columns else None)
    selected_columns = [name for name in (item_column, score_column, pvalue_column) if name]
    if not selected_columns:
        return None
    selected_indexes = [columns.index(name) for name in selected_columns]
    preview_rows: list[list[str]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        preview_rows.append([str(row[index]) if index < len(row) else "" for index in selected_indexes])
    pretty_columns = [
        "Perturbation target" if name == item_column else ("p-value vs random background" if name == pvalue_column else name)
        for name in selected_columns
    ]
    return {
        "columns": pretty_columns,
        "rows": preview_rows,
    }


def _custom_perturbation_form_context(custom_perturbation_backend, job: dict) -> tuple[dict[str, Any], str | None]:
    try:
        return custom_perturbation_backend.build_form_context(job["output_path"]), None
    except FileNotFoundError as exc:
        return _empty_custom_perturbation_form_context(error=str(exc))


def _empty_custom_perturbation_form_context(*, error: str | None = None) -> tuple[dict[str, Any], str | None]:
    return (
        {
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
            "track_options": [],
            "gene_limit": 50,
            "default_direction_mode": "two_sided",
            "background_direction_mode": "two_sided",
            "background_note": "Saved analysis background: two_sided. One-sided and two-sided perturbations use different random backgrounds.",
        },
        error,
    )


def _resolve_perturbation_panel_mode(form_data: Mapping[str, Any] | None = None) -> str:
    if not form_data:
        return "precomputed"
    value = str(form_data.get("pert_panel_mode") or "precomputed").strip().lower()
    if value not in {"precomputed", "custom"}:
        return "precomputed"
    return value


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "image"
    if suffix == ".csv":
        return "csv"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in TEXT_PREVIEW_SUFFIXES:
        return "text"
    return "file"


def _csv_preview(path: Path, *, rows: int = 10) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return {
        "columns": [str(column) for column in frame.columns.tolist()],
        "rows": frame.head(rows).fillna("").astype(str).values.tolist(),
        "row_count": int(len(frame)),
    }


def _safe_read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _humanize_visualization_name(name: str) -> str:
    if name in VISUALIZATION_ENTRY_LABELS:
        return VISUALIZATION_ENTRY_LABELS[name]
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else str(name)


def _group_visualization_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(entries)
    groups: list[dict[str, Any]] = []
    for spec in VISUALIZATION_GROUP_SPECS:
        grouped_entries = [entry for entry in remaining if entry["name"] in spec["members"]]
        if not grouped_entries:
            continue
        label = spec["label"]
        if (
            spec["name"] == "temporal_dynamics_overview"
            and len(grouped_entries) == 1
            and grouped_entries[0]["name"] == "stage_umap_grid"
        ):
            label = _humanize_visualization_name("stage_umap_grid")
        groups.append(
            {
                "name": spec["name"],
                "label": label,
                "entries": grouped_entries,
            }
        )
        grouped_names = {entry["name"] for entry in grouped_entries}
        remaining = [entry for entry in remaining if entry["name"] not in grouped_names]

    for entry in remaining:
        groups.append(
            {
                "name": entry["name"],
                "label": _humanize_visualization_name(entry["name"]),
                "entries": [entry],
            }
        )
    return groups


def _versioned_file_url(path: Path) -> str:
    try:
        version = f"&v={int(path.stat().st_mtime_ns)}"
    except OSError:
        version = ""
    return f"/files?path={quote(str(path), safe='')}{version}"


def _visualization_context(job: dict) -> dict[str, Any]:
    manifest = ((job.get("visualizations") or {}).get("manifest")) or {}
    successes = manifest.get("successes", []) or []
    skipped_names = {
        "pathway_perturbation_table",
        "drug_perturbation_table",
        "single_gene_perturbation_table",
        "gene_combo_perturbation_table",
    }
    static_pdf_route_base = f"/analysis/{job['job_id']}/visualizations"

    items = []
    for success in successes:
        name = str(success.get("name", "artifact"))
        if name in skipped_names:
            continue
        artifacts = []
        artifact_keys = ("file", "table", "matrix", "nodes", "edges")
        if name == "temporal_dynamics_graph":
            artifact_keys = ("file",)
        elif name == "dynamic_marker_trajectories":
            artifact_keys = ("file",)
        elif name in {"stage_composition", "hierarchical_marker_heatmap"}:
            artifact_keys = ("file",)
        for key in artifact_keys:
            raw = success.get(key)
            if not raw:
                continue
            path = Path(raw)
            artifact_type = _artifact_type(path)
            publication_download = None
            if name in STATIC_PDF_EXPORTS and key == "file":
                publication_download = f"{static_pdf_route_base}/{name}/publication.pdf"
            elif artifact_type == "pdf":
                publication_download = _versioned_file_url(path)
            artifacts.append(
                {
                    "label": key,
                    "path": str(path),
                    "url": _versioned_file_url(path),
                    "type": artifact_type,
                    "preview": _csv_preview(path) if path.suffix.lower() == ".csv" else None,
                    "downloads": [],
                    "publication_download": publication_download,
                }
            )
        for raw in success.get("files", []) or []:
            path = Path(raw)
            artifacts.append(
                {
                    "label": "file",
                    "path": str(path),
                    "url": _versioned_file_url(path),
                    "type": _artifact_type(path),
                    "preview": None,
                    "downloads": [],
                    "publication_download": _versioned_file_url(path) if path.suffix.lower() == ".pdf" else None,
                }
            )
        if artifacts:
            downloads = artifacts[0]["downloads"]
            if name == "pathway_ranked_bar":
                for paired in successes:
                    if paired.get("name") == "pathway_perturbation_table" and paired.get("table"):
                        downloads.append(
                            {
                                "label": "table",
                                "path": str(Path(paired["table"])),
                                "url": _versioned_file_url(Path(paired["table"])),
                                "type": _artifact_type(Path(paired["table"])),
                            }
                        )
            elif name == "drug_ranked_bar":
                for paired in successes:
                    if paired.get("name") == "drug_perturbation_table" and paired.get("table"):
                        downloads.append(
                            {
                                "label": "table",
                                "path": str(Path(paired["table"])),
                                "url": _versioned_file_url(Path(paired["table"])),
                                "type": _artifact_type(Path(paired["table"])),
                            }
                        )
            elif name == "single_gene_ranked_bar":
                for paired in successes:
                    if paired.get("name") == "single_gene_perturbation_table" and paired.get("table"):
                        downloads.append(
                            {
                                "label": "table",
                                "path": str(Path(paired["table"])),
                                "url": _versioned_file_url(Path(paired["table"])),
                                "type": _artifact_type(Path(paired["table"])),
                            }
                        )
            elif name == "gene_combo_ranked_bar":
                for paired in successes:
                    if paired.get("name") == "gene_combo_perturbation_table" and paired.get("table"):
                        downloads.append(
                            {
                                "label": "table",
                                "path": str(Path(paired["table"])),
                                "url": _versioned_file_url(Path(paired["table"])),
                                "type": _artifact_type(Path(paired["table"])),
                            }
                        )
            elif name == "hierarchical_marker_heatmap":
                for extra_key in ("table", "matrix"):
                    if success.get(extra_key):
                        downloads.append(
                            {
                                "label": extra_key,
                                "path": str(Path(success[extra_key])),
                                "url": _versioned_file_url(Path(success[extra_key])),
                                "type": _artifact_type(Path(success[extra_key])),
                            }
                        )
        items.append({"name": name, "title": _humanize_visualization_name(name), "meta": success, "artifacts": artifacts})

    idrem_dir = Path(job["output_path"]) / "idrem"
    idrem_files = []
    if idrem_dir.exists():
        for path in sorted(idrem_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".html", ".htm", ".pdf", ".png", ".svg"}:
                idrem_files.append({"path": str(path), "type": _artifact_type(path), "name": path.name})

    temporal_overview = [
        next(entry for entry in items if entry["name"] == name)
        for name in TEMPORAL_OVERVIEW_ENTRY_ORDER
        if any(entry["name"] == name for entry in items)
    ]
    stage_umap = next((entry for entry in temporal_overview if entry["name"] == "stage_umap_grid"), None)
    temporal_graph = next((entry for entry in temporal_overview if entry["name"] == "temporal_dynamics_graph"), None)
    stage_composition = next((entry for entry in temporal_overview if entry["name"] == "stage_composition"), None)
    temporal_overview_names = {entry["name"] for entry in temporal_overview}
    grouped_items = [entry for entry in items if entry["name"] not in temporal_overview_names]

    return {
        "manifest": manifest,
        "entries": items,
        "temporal_overview": temporal_overview,
        "stage_umap": stage_umap,
        "temporal_graph": temporal_graph,
        "stage_composition": stage_composition,
        "groups": _group_visualization_entries(grouped_items),
        "failures": manifest.get("failures", []),
        "idrem_files": idrem_files,
    }


def _share_bundle_context(job: dict) -> dict[str, Any]:
    output_path = Path(job["output_path"])
    bundle_dir = output_path / "share_bundle"
    manifest_path = bundle_dir / "manifest.json"
    manifest = _safe_read_manifest(manifest_path)
    manifest_zip_path = Path(str(manifest.get("zip_path") or "")).resolve() if isinstance(manifest, dict) and manifest.get("zip_path") else None
    zip_path = manifest_zip_path if manifest_zip_path and manifest_zip_path.exists() else (output_path / "share_bundle.zip")
    return {
        "dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "zip_path": str(zip_path),
        "exists": bundle_dir.exists(),
        "manifest": manifest,
        "zip_name": zip_path.name if zip_path.exists() else "",
        "download_url": f"/files?path={quote(str(zip_path), safe='')}" if zip_path.exists() else "",
    }


class DirectTaskRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._active_visualization_tasks: dict[str, str] = {}
        self._active_share_bundle_tasks: dict[str, str] = {}

    @staticmethod
    def _read_visualization_manifest(analysis_dir: Path) -> dict[str, Any] | None:
        manifest_path = analysis_dir / "visualizations" / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def start_visualization_task(self, *, job_id: str, analysis_dir: Path, force: bool = False) -> dict[str, Any]:
        analysis_dir = analysis_dir.expanduser().resolve()
        existing_manifest = None if force else self._read_visualization_manifest(analysis_dir)
        if existing_manifest is not None:
            task_id = uuid.uuid4().hex
            payload = {
                "task_id": task_id,
                "kind": "visualization_generation",
                "job_id": job_id,
                "analysis_dir": str(analysis_dir),
                "status": "succeeded",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result": existing_manifest,
                "error": None,
                "reused_existing": True,
            }
            with self._lock:
                self._tasks[task_id] = payload
            return copy.deepcopy(payload)

        with self._lock:
            active_task_id = self._active_visualization_tasks.get(job_id)
            active_task = self._tasks.get(active_task_id) if active_task_id else None
            if active_task and active_task.get("status") in {"queued", "running"}:
                return copy.deepcopy(active_task)

        task_id = uuid.uuid4().hex
        payload = {
            "task_id": task_id,
            "kind": "visualization_generation",
            "job_id": job_id,
            "analysis_dir": str(analysis_dir),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
            "reused_existing": False,
        }
        with self._lock:
            self._tasks[task_id] = payload
            self._active_visualization_tasks[job_id] = task_id

        def runner() -> None:
            self._update(task_id, status="running")
            try:
                manifest = generate_analysis_visualizations(analysis_dir)
            except Exception as exc:
                self._update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}")
                return
            self._update(task_id, status="succeeded", result=manifest)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return copy.deepcopy(payload)

    @staticmethod
    def _read_share_bundle_manifest(analysis_dir: Path) -> dict[str, Any] | None:
        manifest_path = analysis_dir / "share_bundle" / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def start_share_bundle_task(
        self,
        *,
        job_id: str,
        analysis_dir: Path,
        analysis_label: str,
        force: bool = False,
    ) -> dict[str, Any]:
        analysis_dir = analysis_dir.expanduser().resolve()
        existing_manifest = None if force else self._read_share_bundle_manifest(analysis_dir)
        if existing_manifest is not None:
            task_id = uuid.uuid4().hex
            payload = {
                "task_id": task_id,
                "kind": "share_bundle_export",
                "job_id": job_id,
                "analysis_dir": str(analysis_dir),
                "status": "succeeded",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result": existing_manifest,
                "error": None,
                "reused_existing": True,
            }
            with self._lock:
                self._tasks[task_id] = payload
            return copy.deepcopy(payload)

        with self._lock:
            active_task_id = self._active_share_bundle_tasks.get(job_id)
            active_task = self._tasks.get(active_task_id) if active_task_id else None
            if active_task and active_task.get("status") in {"queued", "running"}:
                return copy.deepcopy(active_task)

        task_id = uuid.uuid4().hex
        payload = {
            "task_id": task_id,
            "kind": "share_bundle_export",
            "job_id": job_id,
            "analysis_dir": str(analysis_dir),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
            "reused_existing": False,
        }
        with self._lock:
            self._tasks[task_id] = payload
            self._active_share_bundle_tasks[job_id] = task_id

        def runner() -> None:
            self._update(task_id, status="running")
            try:
                manifest = export_analysis_share_bundle(
                    analysis_dir,
                    force=force,
                    analysis_label=analysis_label,
                )
            except Exception as exc:
                self._update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}")
                return
            self._update(task_id, status="succeeded", result=manifest)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return copy.deepcopy(payload)

    def _update(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.update(updates)
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            job_id = str(task.get("job_id", ""))
            active_map = None
            if task.get("kind") == "visualization_generation":
                active_map = self._active_visualization_tasks
            elif task.get("kind") == "share_bundle_export":
                active_map = self._active_share_bundle_tasks
            if active_map is not None:
                if task.get("status") in {"queued", "running"}:
                    active_map[job_id] = task_id
                elif active_map.get(job_id) == task_id:
                    active_map.pop(job_id, None)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return copy.deepcopy(task) if task is not None else None


def create_monitor_app(
    *,
    skill_root: Path,
    repo_root: Path,
    skill_dir: Path,
    job_roots: Iterable[Path] | None = None,
    data_roots: Iterable[Path] | None = None,
) -> FastAPI:
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root, job_roots=job_roots)
    job_backend = _load_job_backend(skill_dir)
    custom_perturbation_backend = _load_custom_perturbation_backend()
    explorer_service = AnalysisExplorerService(skill_root=skill_root)
    allowed_roots = _unique_paths(list(data_roots or []) + list(job_roots or []) + [repo_root, skill_root])
    direct_tasks = DirectTaskRegistry()

    templates_dir = skill_dir / "web" / "templates"
    static_dir = skill_dir / "web" / "static"

    app = FastAPI(title="UNAGI Console", version="1.1.0")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))
    templates.env.globals["active_statuses"] = ACTIVE_STATUSES
    templates.env.globals["allowed_roots"] = [str(path) for path in allowed_roots]
    def _static_url(request: Request, path: str) -> str:
        asset_path = static_dir / str(path)
        try:
            version = str(asset_path.stat().st_mtime_ns)
        except FileNotFoundError:
            version = "0"
        return f"{request.url_for('static', path=path)}?v={version}"
    templates.env.globals["static_url"] = _static_url
    templates.env.filters["urlquote"] = lambda value: quote(str(value), safe="")
    templates.env.filters["querystring"] = lambda value: urlencode(value, doseq=True)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, status: str | None = None, kind: str | None = None):
        all_jobs = service.list_jobs()
        jobs = service.list_jobs(status=status)
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "jobs": jobs,
                "dashboard_view": _build_dashboard_view(all_jobs, jobs),
                "counts": _summary_counts(jobs),
                "status_filter": status or "",
            },
        )

    @app.get("/experiments/new", response_class=HTMLResponse)
    def experiment_new(request: Request, template_path: str | None = None):
        template_payload, resolved_template = job_backend.load_training_template(repo_root, template_path or None)
        return templates.TemplateResponse(
            "experiments_new.html",
            {
                "request": request,
                "allowed_roots": [str(path) for path in allowed_roots],
                "template_path": str(resolved_template),
                "initial": _base_training_initial(template_payload, template_path=str(resolved_template)),
                "advanced_yaml": _yaml_text(template_payload),
            },
        )

    @app.post("/api/training/validate")
    async def training_validate(request: Request):
        form_data = await _parse_request_form(request)
        payload: dict[str, Any] = {}
        try:
            payload, _meta = _training_payload_from_form(
                job_backend,
                form_data,
                repo_root=repo_root,
                allowed_roots=allowed_roots,
            )
            result = run_workflow_from_payload(payload, validate_only=True, persist_updates=False)
            effective_config = copy.deepcopy(payload)
            effective_config.setdefault("run", {})["idrem_dir"] = result["idrem_dir"]
            return JSONResponse(
                {
                    "ok": True,
                    "normalized_data_dir": result["data_dir"],
                    "resolved_idrem_dir": result["idrem_dir"],
                    "warnings": [],
                    "errors": [],
                    "effective_config": effective_config,
                }
            )
        except (PreflightError, ValueError, FileNotFoundError) as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "normalized_data_dir": None,
                    "resolved_idrem_dir": None,
                    "warnings": [],
                    "errors": [str(exc)],
                    "effective_config": payload or None,
                },
                status_code=http_status.HTTP_400_BAD_REQUEST,
            )

    @app.post("/api/training")
    async def training_submit(request: Request):
        form_data = await _parse_request_form(request)
        try:
            payload, meta = _training_payload_from_form(
                job_backend,
                form_data,
                repo_root=repo_root,
                allowed_roots=allowed_roots,
            )
            job = job_backend.create_job_from_config_payload(
                repo_root=repo_root,
                config_payload=payload,
                source_config_path=meta["template_path"] or None,
                max_retries=int(form_data.get("max_retries") or 0),
                retry_delay_sec=int(form_data.get("retry_delay_sec") or 60),
                job_name=meta["job_name"],
            )
            return JSONResponse(
                {
                    "status": "submitted",
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "detail_url": f"/jobs/{job['job_id']}",
                },
                status_code=http_status.HTTP_201_CREATED,
            )
        except (PreflightError, ValueError, FileNotFoundError) as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=http_status.HTTP_400_BAD_REQUEST)

    @app.get("/partials/dashboard", response_class=HTMLResponse)
    def dashboard_partial(request: Request, status: str | None = None, kind: str | None = None):
        all_jobs = service.list_jobs()
        jobs = service.list_jobs(status=status)
        return templates.TemplateResponse(
            "partials/dashboard_panel.html",
            {
                "request": request,
                "jobs": jobs,
                "dashboard_view": _build_dashboard_view(all_jobs, jobs),
                "counts": _summary_counts(jobs),
                "status_filter": status or "",
            },
        )

    @app.get("/partials/system", response_class=HTMLResponse)
    def system_partial(request: Request):
        snapshot = service.get_system_snapshot()
        return templates.TemplateResponse(
            "partials/system_panel.html",
            {"request": request, "system": snapshot},
        )

    @app.get("/partials/path-picker", response_class=HTMLResponse)
    def path_picker_partial(
        request: Request,
        target: str,
        path: str | None = None,
        kind: str = "path",
    ):
        current = _safe_path(path, allowed_roots) if path else allowed_roots[0]
        if current.is_file():
            current = current.parent
        entries = []
        for child in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            selectable = _selection_kind_matches(child, kind)
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "selectable": selectable,
                }
            )
        parent_path = None
        for root in allowed_roots:
            try:
                current.relative_to(root)
                if current != root:
                    parent_path = str(current.parent)
                break
            except ValueError:
                continue
        return templates.TemplateResponse(
            "partials/path_picker.html",
            {
                "request": request,
                "target": target,
                "kind": kind,
                "path": str(current),
                "parent_path": parent_path,
                "entries": entries,
            },
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: str):
        try:
            job = service.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse(
            "job_detail.html",
            {
                "request": request,
                "job": job,
                "log_tail": service.get_log_tail(job_id),
            },
        )

    @app.get("/jobs/{job_id}/analysis/new", response_class=HTMLResponse)
    def analysis_new(request: Request, job_id: str, iteration: int):
        job = service.get_job(job_id)
        if job["job_kind"] != "train":
            raise HTTPException(status_code=400, detail="Analysis can only be launched from a training job")
        valid_iterations = {item["iteration"] for item in job.get("finished_iterations", [])}
        if int(iteration) not in valid_iterations:
            raise HTTPException(status_code=404, detail=f"Finished iteration {iteration} was not found")
        template_payload, resolved_template = job_backend._load_analysis_template(repo_root, None)
        return templates.TemplateResponse(
            "analysis_new.html",
            {
                "request": request,
                "job": job,
                "iteration": int(iteration),
                "allowed_roots": [str(path) for path in allowed_roots],
                "template_path": str(resolved_template) if resolved_template else "",
                "initial": _base_analysis_initial(
                    template_payload,
                    template_path=str(resolved_template) if resolved_template else "",
                    iteration=int(iteration),
                ),
                "advanced_yaml": _yaml_text(template_payload),
            },
        )

    @app.post("/api/analysis")
    async def analysis_submit(request: Request):
        form_data = await _parse_request_form(request)
        training_job_id = str(form_data.get("training_job_id") or "").strip()
        if not training_job_id:
            raise HTTPException(status_code=400, detail="training_job_id is required")
        iteration = int(form_data.get("iteration") or 0)
        try:
            payload, meta = _analysis_payload_from_form(
                job_backend,
                form_data,
                repo_root=repo_root,
                allowed_roots=allowed_roots,
                iteration=iteration,
            )
            job = job_backend.create_analysis_job_from_params(
                repo_root=repo_root,
                training_job_id=training_job_id,
                iteration=iteration,
                params_payload=payload,
                analysis_config_source=meta["template_path"] or None,
                max_retries=int(form_data.get("max_retries") or 0),
                retry_delay_sec=int(form_data.get("retry_delay_sec") or 60),
                job_name=meta["job_name"],
            )
            return JSONResponse(
                {
                    "status": "submitted",
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "detail_url": f"/jobs/{job['job_id']}",
                },
                status_code=http_status.HTTP_201_CREATED,
            )
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=http_status.HTTP_400_BAD_REQUEST)

    @app.post("/api/analysis/{job_id}/perturbations/validate")
    async def analysis_perturbation_validate(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Custom perturbation is only available for analysis jobs")
        form_data = await _parse_request_form(request)
        raw_payload = _custom_perturbation_payload_from_form(form_data)
        try:
            result = custom_perturbation_backend.validate_custom_perturbation_payload(
                analysis_dir=job["output_path"],
                analysis_job_id=job_id,
                analysis_iteration=int(job.get("analysis_iteration") or 0),
                raw_payload=raw_payload,
            )
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse(
                {"ok": False, "errors": [str(exc)], "warnings": []},
                status_code=http_status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse(
            {
                "ok": True,
                "errors": [],
                "warnings": result["warnings"],
                "estimated_scope_cell_count": result["estimated_scope_cell_count"],
                "analysis_n_obs": result["analysis_n_obs"],
                "resolved_execution_mode": result["resolved_execution_mode"],
                "resolved_execution_backend": result.get("resolved_execution_backend"),
                "cpu_workers": result.get("cpu_workers"),
                "queue_key": result["queue_key"],
                "normalized": result["normalized"],
            }
        )

    @app.post("/api/analysis/{job_id}/perturbations")
    async def analysis_perturbation_submit(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Custom perturbation is only available for analysis jobs")
        form_data = await _parse_request_form(request)
        raw_payload = _custom_perturbation_payload_from_form(form_data)
        try:
            perturb_job = job_backend.create_custom_perturbation_job_from_file(
                repo_root=repo_root,
                analysis_job_file_path=job["job_file_path"],
                analysis_job_id=job_id,
                request_payload=raw_payload,
                max_retries=int(form_data.get("max_retries") or 0),
                retry_delay_sec=int(form_data.get("retry_delay_sec") or 60),
            )
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse(
                {"ok": False, "errors": [str(exc)]},
                status_code=http_status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse(
            {
                "status": "submitted",
                "job_id": perturb_job["job_id"],
                "job_name": perturb_job["job_name"],
                "detail_url": f"/analysis/{job_id}/perturbations/{perturb_job['job_id']}",
            },
            status_code=http_status.HTTP_201_CREATED,
        )

    @app.get("/api/analysis/{job_id}/perturbations/suggest")
    def analysis_perturbation_suggest(job_id: str, kind: str, q: str = "", limit: int = 12):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Custom perturbation is only available for analysis jobs")
        try:
            suggestions = custom_perturbation_backend.autocomplete_suggestions(
                job["output_path"],
                kind=kind,
                query=q,
                limit=limit,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {
                "kind": str(kind),
                "query": str(q),
                "suggestions": suggestions,
            }
        )

    @app.get("/analysis/{job_id}/visualizations", response_class=HTMLResponse)
    def analysis_visualizations(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Visualizations are only available for analysis jobs")
        return templates.TemplateResponse(
            "analysis_visualizations.html",
            {
                "request": request,
                "job": job,
                "visualization": _visualization_context(job),
                "share_bundle": _share_bundle_context(job),
            },
        )

    @app.get("/analysis/{job_id}/perturbations/{perturb_job_id}", response_class=HTMLResponse)
    def analysis_custom_perturbation_detail(request: Request, job_id: str, perturb_job_id: str):
        parent_job = service.get_job(job_id)
        perturb_job = service.get_job(perturb_job_id)
        if parent_job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Parent job must be an analysis job")
        if perturb_job["job_kind"] != "custom_perturbation":
            raise HTTPException(status_code=400, detail="Requested job is not a custom perturbation result")
        if str(perturb_job.get("parent_job_id") or "") != job_id:
            raise HTTPException(status_code=404, detail="Custom perturbation job does not belong to this analysis")

        result = None
        result_error = None
        result_preview = None
        if Path(perturb_job["output_path"]).exists():
            try:
                result = custom_perturbation_backend.load_result_bundle(perturb_job["output_path"])
                stale_error = _stale_custom_perturbation_error(result)
                if stale_error:
                    result_error = stale_error
                    result = None
                else:
                    result_preview = _custom_perturbation_score_preview(result)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                result_error = str(exc)
        return templates.TemplateResponse(
            "analysis_custom_perturbation.html",
            {
                "request": request,
                "analysis_job": parent_job,
                "job": perturb_job,
                "result": result,
                "result_error": result_error,
                "result_preview": result_preview,
            },
        )

    @app.get("/api/analysis/{job_id}/perturbations/{perturb_job_id}")
    def analysis_custom_perturbation_api(job_id: str, perturb_job_id: str):
        parent_job = service.get_job(job_id)
        perturb_job = service.get_job(perturb_job_id)
        if parent_job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Parent job must be an analysis job")
        if perturb_job["job_kind"] != "custom_perturbation":
            raise HTTPException(status_code=400, detail="Requested job is not a custom perturbation result")
        if str(perturb_job.get("parent_job_id") or "") != job_id:
            raise HTTPException(status_code=404, detail="Custom perturbation job does not belong to this analysis")

        result = None
        if Path(perturb_job["output_path"]).exists():
            try:
                result = custom_perturbation_backend.load_result_bundle(perturb_job["output_path"])
                stale_error = _stale_custom_perturbation_error(result)
                if stale_error:
                    raise HTTPException(status_code=409, detail=stale_error)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"job": perturb_job, "result": result})

    @app.get("/analysis/{job_id}/idrem/track-popup", response_class=HTMLResponse)
    def analysis_idrem_track_popup(
        request: Request,
        job_id: str,
        track_key: str,
        track_label: str | None = None,
    ):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Track iDREM summaries are only available for analysis jobs")
        query = {"track_key": track_key}
        if track_label:
            query["track_label"] = track_label
        return templates.TemplateResponse(
            "analysis_idrem_track_popup.html",
            {
                "request": request,
                "job": job,
                "track_key": track_key,
                "track_label": track_label or "",
                "track_url": f"/api/analysis/{job_id}/idrem/track?{urlencode(query)}",
            },
        )

    @app.get("/analysis/{job_id}/visualizations/{figure_name}/publication.pdf")
    def analysis_visualization_publication_pdf(job_id: str, figure_name: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Publication exports are only available for analysis jobs")
        try:
            pdf_bytes, filename = export_static_publication_pdf(job["output_path"], figure_name)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/partials/analysis/{job_id}/explorer", response_class=HTMLResponse)
    def analysis_explorer_partial(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Interactive explorer is only available for analysis jobs")
        perturbation_panel_mode = _resolve_perturbation_panel_mode(request.query_params)
        selected_custom_job_id = _selected_custom_perturbation_job_id(request.query_params)
        try:
            explorer = explorer_service.build_view(job["output_path"], form_data=request.query_params)
        except FileNotFoundError as exc:
            explorer = {
                "analysis_dir": job["output_path"],
                "active_mode": "custom_genes",
                "mode_options": [],
                "metadata": {"gene_count": 0, "gene_limit": 12, "dataset_path": None, "attribute_path": None},
                "umap": {"available": False, "reason": str(exc), "color_options": [], "color_by": ""},
                "hierarchical": {"available": False, "reason": str(exc)},
                "dynamic_markers": {"available": False, "reason": str(exc)},
                "custom_genes": {"available": False, "reason": str(exc), "gene_text": "", "track_options": []},
                "perturbation": {"available": False, "reason": str(exc)},
                "result": {"status": "error", "title": "Interactive Explorer", "message": str(exc)},
            }
        if explorer.get("active_mode") == "perturbation" and perturbation_panel_mode == "custom":
            custom_perturbation_form, custom_perturbation_form_error = _custom_perturbation_form_context(
                custom_perturbation_backend,
                job,
            )
            custom_detail_context = _custom_perturbation_detail_context(
                service=service,
                custom_perturbation_backend=custom_perturbation_backend,
                job_backend=job_backend,
                repo_root=repo_root,
                analysis_job=job,
                selected_job_id=selected_custom_job_id,
            )
        else:
            custom_perturbation_form, custom_perturbation_form_error = _empty_custom_perturbation_form_context()
            custom_detail_context = {
                "selected_custom_perturbation_job": None,
                "selected_custom_perturbation_result": None,
                "selected_custom_perturbation_error": None,
                "selected_custom_perturbation_preview": None,
            }
        return templates.TemplateResponse(
            "partials/analysis_explorer.html",
            {
                "request": request,
                "job": job,
                "explorer": explorer,
                "perturbation_panel_mode": perturbation_panel_mode,
                "custom_perturbation_form": custom_perturbation_form,
                "custom_perturbation_form_error": custom_perturbation_form_error,
                "custom_perturbation_jobs": _custom_perturbation_jobs(job, job_backend=job_backend, repo_root=repo_root),
                **custom_detail_context,
            },
        )

    @app.post("/partials/analysis/{job_id}/explorer/render", response_class=HTMLResponse)
    async def analysis_explorer_render_partial(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Interactive explorer is only available for analysis jobs")
        form_data = await _parse_request_form(request)
        perturbation_panel_mode = _resolve_perturbation_panel_mode(form_data)
        selected_custom_job_id = _selected_custom_perturbation_job_id(form_data)
        try:
            explorer = explorer_service.build_view(job["output_path"], form_data=form_data)
        except FileNotFoundError as exc:
            explorer = {
                "analysis_dir": job["output_path"],
                "active_mode": "custom_genes",
                "mode_options": [],
                "metadata": {"gene_count": 0, "gene_limit": 12, "dataset_path": None, "attribute_path": None},
                "umap": {"available": False, "reason": str(exc), "color_options": [], "color_by": ""},
                "hierarchical": {"available": False, "reason": str(exc)},
                "dynamic_markers": {"available": False, "reason": str(exc)},
                "custom_genes": {"available": False, "reason": str(exc), "gene_text": "", "track_options": []},
                "perturbation": {"available": False, "reason": str(exc)},
                "result": {"status": "error", "title": "Interactive Explorer", "message": str(exc)},
            }
        if explorer.get("active_mode") == "perturbation" and perturbation_panel_mode == "custom":
            custom_perturbation_form, custom_perturbation_form_error = _custom_perturbation_form_context(
                custom_perturbation_backend,
                job,
            )
            custom_detail_context = _custom_perturbation_detail_context(
                service=service,
                custom_perturbation_backend=custom_perturbation_backend,
                job_backend=job_backend,
                repo_root=repo_root,
                analysis_job=job,
                selected_job_id=selected_custom_job_id,
            )
        else:
            custom_perturbation_form, custom_perturbation_form_error = _empty_custom_perturbation_form_context()
            custom_detail_context = {
                "selected_custom_perturbation_job": None,
                "selected_custom_perturbation_result": None,
                "selected_custom_perturbation_error": None,
                "selected_custom_perturbation_preview": None,
            }
        return templates.TemplateResponse(
            "partials/analysis_explorer.html",
            {
                "request": request,
                "job": job,
                "explorer": explorer,
                "perturbation_panel_mode": perturbation_panel_mode,
                "custom_perturbation_form": custom_perturbation_form,
                "custom_perturbation_form_error": custom_perturbation_form_error,
                "custom_perturbation_jobs": _custom_perturbation_jobs(job, job_backend=job_backend, repo_root=repo_root),
                **custom_detail_context,
            },
        )

    @app.get("/analysis/{job_id}/explorer/publication.pdf")
    def analysis_explorer_publication_pdf(
        request: Request,
        job_id: str,
        mode: str,
        figure_key: str | None = None,
    ):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Publication exports are only available for analysis jobs")
        try:
            pdf_bytes, filename = explorer_service.export_publication_pdf(job["output_path"], form_data=request.query_params)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/analysis/{job_id}/explorer/table.csv")
    def analysis_explorer_table_csv(request: Request, job_id: str, mode: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Explorer table exports are only available for analysis jobs")
        try:
            csv_bytes, filename = explorer_service.export_table_csv(job["output_path"], form_data=request.query_params)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/analysis/{job_id}/umap")
    def analysis_umap_api(
        job_id: str,
        stage: str | None = None,
        color_by: str | None = None,
        gene: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        screen_width: int | None = None,
        screen_height: int | None = None,
        current_render_mode: str | None = None,
    ):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="UMAP data is only available for analysis jobs")
        try:
            payload = explorer_service.get_umap_payload(
                job["output_path"],
                stage=stage,
                color_by=color_by,
                gene=gene,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                screen_width=screen_width,
                screen_height=screen_height,
                current_render_mode=current_render_mode,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/analysis/{job_id}/hierarchical/dendrogram")
    def analysis_hierarchical_dendrogram_api(job_id: str, stage: str, cluster: str | None = None):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Hierarchical explorer data is only available for analysis jobs")
        try:
            payload = explorer_service.get_hierarchical_dendrogram_spec(job["output_path"], stage=stage, cluster=cluster)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/analysis/{job_id}/hierarchical/heatmap")
    def analysis_hierarchical_heatmap_api(
        job_id: str,
        stage: str,
        cluster: str,
        level: str,
        direction: str = "up",
        top_n: int = Query(20, ge=1, le=100),
    ):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Hierarchical explorer data is only available for analysis jobs")
        try:
            payload = explorer_service.get_hierarchical_heatmap_spec(
                job["output_path"],
                stage=stage,
                cluster=cluster,
                level=level,
                direction=direction,
                top_n=top_n,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/analysis/{job_id}/idrem/track")
    def analysis_idrem_track_api(job_id: str, track_key: str, track_label: str | None = None):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="iDREM summaries are only available for analysis jobs")
        try:
            payload = explorer_service.get_idrem_track_summary(
                job["output_path"],
                track_key=track_key,
                track_label=track_label or "",
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if payload.get("original_html_path"):
            payload = dict(payload)
            payload["original_html_url"] = f"/files?path={quote(str(payload['original_html_path']), safe='')}"
        return JSONResponse(payload)

    @app.post("/api/analysis/{job_id}/umap/selection-summary")
    async def analysis_umap_selection_summary_api(request: Request, job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="UMAP selection summaries are only available for analysis jobs")
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Expected JSON body") from exc
        try:
            summary = explorer_service.summarize_umap_selection(
                job["output_path"],
                stage=str(payload.get("stage") or ""),
                color_by=str(payload.get("color_by") or ""),
                cell_ids=[int(cell_id) for cell_id in payload.get("cell_ids") or []],
                gene=str(payload.get("gene") or ""),
                render_mode=str(payload.get("render_mode") or ""),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(summary)

    @app.get("/api/analysis/{job_id}/temporal-graph")
    def analysis_temporal_graph_api(job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Temporal graph data is only available for analysis jobs")
        try:
            payload = explorer_service.get_temporal_graph_payload(job["output_path"])
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/analysis/{job_id}/stage-composition")
    def analysis_stage_composition_api(job_id: str):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Stage composition data is only available for analysis jobs")
        try:
            payload = explorer_service.get_stage_composition_payload(job["output_path"])
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        try:
            job = service.get_job(job_id)
            result = job_backend.cancel_job_from_file(job_file_path=job["job_file_path"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/jobs/{job_id}/resume")
    async def job_resume(request: Request, job_id: str):
        form_data = await _parse_request_form(request)
        try:
            job = service.get_job(job_id)
            result = job_backend.resume_job_from_file(
                job_file_path=job["job_file_path"],
                repo_root=repo_root,
                resume_iteration=int(form_data.get("resume_iteration")) if form_data.get("resume_iteration") else None,
                max_retries=int(form_data.get("max_retries")) if form_data.get("max_retries") else None,
                retry_delay_sec=int(form_data.get("retry_delay_sec")) if form_data.get("retry_delay_sec") else None,
                set_overrides=[str(value) for value in form_data.getlist("set") if str(value).strip()],
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=http_status.HTTP_400_BAD_REQUEST)
        return JSONResponse(result)

    @app.post("/api/jobs/{job_id}/delete")
    def job_delete(job_id: str):
        try:
            job = service.get_job(job_id)
            result = job_backend.delete_job_from_file(job_file_path=job["job_file_path"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            return JSONResponse(
                {
                    "status": "blocked",
                    "job_id": job_id,
                    "blocked_reason": str(exc),
                },
                status_code=http_status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse(result)

    @app.post("/api/jobs/delete")
    async def jobs_delete(request: Request):
        form_data = await _parse_request_form(request)
        job_ids = [str(value).strip() for value in form_data.getlist("job_id") if str(value).strip()]
        if not job_ids:
            return JSONResponse(
                {"status": "invalid", "errors": ["At least one job_id is required"]},
                status_code=http_status.HTTP_400_BAD_REQUEST,
            )
        try:
            jobs = [service.get_job(job_id) for job_id in job_ids]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = job_backend.delete_jobs_from_files(job_file_paths=[job["job_file_path"] for job in jobs])
        return JSONResponse(result)

    @app.post("/api/analysis/{job_id}/visualizations/generate")
    def analysis_visualizations_generate(job_id: str, force: bool = Query(default=False)):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Visualizations are only available for analysis jobs")
        task = direct_tasks.start_visualization_task(
            job_id=job_id,
            analysis_dir=Path(job["output_path"]),
            force=force,
        )
        return JSONResponse(task, status_code=http_status.HTTP_202_ACCEPTED)

    @app.post("/api/analysis/{job_id}/share-bundle/export")
    def analysis_share_bundle_export(job_id: str, force: bool = Query(default=False)):
        job = service.get_job(job_id)
        if job["job_kind"] != "analysis":
            raise HTTPException(status_code=400, detail="Share bundles are only available for analysis jobs")
        task = direct_tasks.start_share_bundle_task(
            job_id=job_id,
            analysis_dir=Path(job["output_path"]),
            analysis_label=str(job.get("job_name") or job_id),
            force=force,
        )
        return JSONResponse(task, status_code=http_status.HTTP_202_ACCEPTED)

    @app.get("/api/direct-tasks/{task_id}")
    def direct_task_api(task_id: str):
        task = direct_tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Unknown task_id")
        return JSONResponse(task)

    @app.get("/partials/jobs/{job_id}/summary", response_class=HTMLResponse)
    def job_summary_partial(request: Request, job_id: str):
        try:
            job = service.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse(
            "partials/job_summary.html",
            {"request": request, "job": job},
        )

    @app.get("/partials/jobs/{job_id}/log", response_class=HTMLResponse)
    def job_log_partial(request: Request, job_id: str, lines: int = 160):
        try:
            job = service.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        log_tail = service.get_log_tail(job_id, lines=lines)
        return templates.TemplateResponse(
            "partials/job_log.html",
            {"request": request, "job": job, "log_tail": log_tail},
        )

    @app.get("/api/jobs")
    def jobs_api(status: str | None = None, kind: str | None = None):
        return JSONResponse(service.list_jobs(status=status, kind=kind))

    @app.get("/api/jobs/{job_id}")
    def job_api(job_id: str):
        try:
            return JSONResponse(service.get_job(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/system")
    def system_api():
        return JSONResponse(service.get_system_snapshot())

    @app.get("/browse", response_class=HTMLResponse)
    def browse(request: Request, path: str = Query(...)):
        target = _safe_path(path, allowed_roots)
        if target.is_file():
            if not _is_text_previewable(target):
                return PlainTextResponse(f"Binary file preview is not supported:\n{target}")
            text = target.read_text(encoding="utf-8", errors="replace")
            return templates.TemplateResponse(
                "browse_file.html",
                {"request": request, "path": str(target), "content": text},
            )

        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                }
            )
        return templates.TemplateResponse(
            "browse_dir.html",
            {"request": request, "path": str(target), "entries": entries},
        )

    @app.get("/files")
    def files(path: str = Query(...)):
        target = _safe_path(path, allowed_roots)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        media_type, _ = mimetypes.guess_type(str(target))
        return FileResponse(str(target), media_type=media_type)

    return app
