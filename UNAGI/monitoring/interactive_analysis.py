from __future__ import annotations

import ast
import base64
import io
import threading
import textwrap
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from .explorer_cache import UmapViewportCache
from .idrem_summary import IdremTrackSummaryCache
from UNAGI.plotting import (
    build_palette,
    extract_dynamic_marker_table,
    extract_hierarchical_marker_table,
    extract_perturbation_table,
    plot_dynamic_marker_heatmap,
    plot_dynamic_marker_trajectories,
    plot_hierarchical_marker_branch_heatmap,
    plot_perturbation_ranked_bar,
    plot_temporal_dynamics_graph,
)

from .visualization_pipeline import INTERACTIVE_UNS_KEYS, _read_analysis_adata
from UNAGI.perturbations.analysis_perturbation import perturbationAnalysis

EXPLORER_MODES = (
    ("umap", "UMAP"),
    ("hierarchical", "HIERARCHICAL MARKERS"),
    ("dynamic_markers", "Dynamic Markers"),
    ("perturbation", "Perturbation"),
)

PERTURBATION_MODE_KEYS = {
    "pathway": "pathway_perturbation_score",
    "drug": "drug_perturbation_score",
    "single_gene": "single_gene_perturbation_score",
    "gene_combo": "two_genes_perturbation_score",
}
PERTURBATION_SCORING_UNS_KEYS = INTERACTIVE_UNS_KEYS + (
    "pathway_perturbation_deltaD",
    "drug_perturbation_deltaD",
    "random_background_perturbation_deltaD",
)

GENE_LIMIT = 12
HIERARCHICAL_DIRECTION_LABELS = {
    "up": "Up Regulated",
    "down": "Down Regulated",
}
CUSTOM_GENE_SCOPE_LABELS = {
    "global_stage": "global stage",
    "progression_track": "temporal track",
}


def _sorted_keys(values: Iterable[Any]) -> list[str]:
    def sort_key(value: Any) -> tuple[int, Any]:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(values, key=sort_key)]


def _ordered_dynamic_directions(values: Iterable[Any]) -> list[str]:
    ordered = _sorted_keys(values)
    preferred = ["increasing", "decreasing"]
    front = [name for name in preferred if name in ordered]
    tail = [name for name in ordered if name not in preferred]
    return front + tail


def _cache_key(path: Path | None) -> tuple[str, int | None, int | None]:
    if path is None:
        return ("", None, None)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None, None)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _analysis_iteration_from_dir(analysis_dir: Path) -> str | None:
    for parent in analysis_dir.parents:
        name = str(parent.name)
        if name.startswith('iter') and name[4:].isdigit():
            return name[4:]
    return None


def _stage_specific_perturbation_files(analysis_dir: Path) -> list[Path]:
    base = Path(analysis_dir) / "stage_specific_perturbation"
    if not base.exists():
        return []
    return sorted(path for path in base.glob("stage_*/*.csv") if path.is_file())


def _stage_specific_perturbation_identity(analysis_dir: Path) -> tuple[Any, ...]:
    payload: list[tuple[str, int, int]] = []
    for path in _stage_specific_perturbation_files(analysis_dir):
        stat = path.stat()
        payload.append((str(path.relative_to(analysis_dir)), stat.st_mtime_ns, stat.st_size))
    return tuple(payload)


def _load_temporal_edges_fallback(analysis_dir: Path) -> dict[str, Any] | None:
    iteration = _analysis_iteration_from_dir(analysis_dir)
    if not iteration:
        return None
    try:
        training_root = analysis_dir.parents[3]
    except IndexError:
        return None
    edges_path = training_root / 'input' / iteration / 'edges.txt'
    if not edges_path.exists():
        return None
    try:
        payload = ast.literal_eval(edges_path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_to_pdf_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _preview_frame(frame: pd.DataFrame, *, rows: int = 18) -> dict[str, Any]:
    if frame is None:
        return {"columns": [], "rows": [], "row_count": 0}
    preview = frame.copy()
    preview = preview.replace({np.nan: ""})
    for column in preview.columns:
        if pd.api.types.is_float_dtype(preview[column]):
            preview[column] = preview[column].round(4)
    return {
        "columns": [str(column) for column in preview.columns.tolist()],
        "rows": preview.head(rows).astype(str).values.tolist(),
        "row_count": int(len(preview)),
    }


def _perturbation_label_column(frame: pd.DataFrame) -> str:
    for column in ("pathways", "compound", "gene"):
        if column in frame.columns:
            return column
    for column in frame.columns:
        if str(column) != "perturbation score":
            return str(column)
    return str(frame.columns[0])


def _wrap_perturbation_plotly_label(value: Any, *, max_line_length: int = 34, max_lines: int = 2) -> str:
    text = str(value).replace("_", " ").replace("/", " / ")
    text = " ".join(text.split())
    if not text:
        return ""
    wrapped = textwrap.wrap(
        text,
        width=max_line_length,
        break_long_words=False,
        break_on_hyphens=True,
    )
    if not wrapped:
        wrapped = textwrap.wrap(
            text,
            width=max_line_length,
            break_long_words=True,
            break_on_hyphens=True,
    )
    if len(wrapped) > max_lines:
        wrapped = [wrapped[0], " ".join(wrapped[1:])]
    return "<br>".join(wrapped[:max_lines])


def _perturbation_plotly_spec(frame: pd.DataFrame, *, title: str) -> dict[str, Any]:
    label_column = _perturbation_label_column(frame)
    score_series = pd.to_numeric(frame["perturbation score"], errors="coerce").fillna(0.0)
    pval_series = (
        pd.to_numeric(frame["pval"], errors="coerce").fillna(np.nan)
        if "pval" in frame.columns
        else pd.Series([np.nan] * len(frame))
    )
    qval_series = (
        pd.to_numeric(frame["pval_adjusted"], errors="coerce").fillna(np.nan)
        if "pval_adjusted" in frame.columns
        else pd.Series([np.nan] * len(frame))
    )
    labels = frame[label_column].astype(str).tolist()[::-1]
    wrapped_labels = [_wrap_perturbation_plotly_label(label) for label in labels]
    line_counts = [max(1, label.count("<br>") + 1) for label in wrapped_labels]
    max_line_count = max(line_counts, default=1)
    max_wrapped_chars = max(
        (max((len(part) for part in label.split("<br>")), default=0) for label in wrapped_labels),
        default=0,
    )
    margin_left = min(460, max(340, int(190 + 7.0 * max_wrapped_chars)))
    scores = score_series.astype(float).tolist()[::-1]
    pvals = pval_series.astype(float).tolist()[::-1]
    qvals = qval_series.astype(float).tolist()[::-1]
    text: list[str] = []
    hover_text = []
    for label, score, pval, qval in zip(labels, scores, pvals, qvals):
        lines = [f"{label_column}: {label}", f"perturbation score: {score:.4f}"]
        visible_lines = [f"score {score:.4f}"]
        if np.isfinite(pval):
            lines.append(f"p-value: {pval:.4g}")
            visible_lines.append(f"p {pval:.4g}")
        if np.isfinite(qval):
            lines.append(f"adjusted p-value: {qval:.4g}")
            visible_lines.append(f"adj.p {qval:.4g}")
        hover_text.append("<br>".join(lines))
        text.append("<br>".join(visible_lines))
    height = max(420, int(58 * max(len(labels), 1) + 110))
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": scores,
                "y": wrapped_labels,
                "text": text,
                "textposition": "outside",
                "marker": {"color": "#9d5c3f"},
                "hovertemplate": "%{customdata}<extra></extra>",
                "customdata": hover_text,
                "cliponaxis": False,
            }
        ],
        "layout": {
            "title": {"text": title},
            "height": height,
            "margin": {"l": margin_left, "r": 96, "t": 56, "b": 56},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "title": {"text": "perturbation score"},
                "automargin": True,
                "domain": [0.0, 1.0],
            },
            "yaxis": {
                "automargin": False,
                "ticklabeloverflow": "allow",
                "tickfont": {"size": 9.5 if max_line_count >= 2 else 10.5},
            },
        },
        "config": {"responsive": True, "displayModeBar": False},
    }


def _hierarchical_marker_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    export_frame = frame.copy()
    if "Gene" in export_frame.columns:
        pass
    elif "gene" in export_frame.columns:
        export_frame = export_frame.rename(columns={"gene": "Gene"})
    else:
        export_frame = export_frame.rename_axis("Gene").reset_index()
    ordered_columns = ["Gene"] + [column for column in export_frame.columns if str(column) != "Gene"]
    return export_frame.loc[:, ordered_columns]


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _int_value(value: Any, *, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _form_list(form_data: Mapping[str, Any] | None, key: str) -> list[str]:
    if form_data is None:
        return []
    getlist = getattr(form_data, "getlist", None)
    if callable(getlist):
        return [str(item) for item in getlist(key) if str(item).strip()]
    value = form_data.get(key)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _resolve_string(options: Iterable[str], raw: Any, *, default: str | None = None) -> str:
    valid = [str(option) for option in options]
    if not valid:
        return default or ""
    text = str(raw) if raw not in (None, "") else ""
    if text in valid:
        return text
    return default if default in valid else valid[0]


def _split_gene_text(raw_text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in str(raw_text).replace("\n", ",").split(","):
        for token in chunk.strip().split():
            if token:
                tokens.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _build_gene_lookup(adata) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact_names = [str(name) for name in adata.var_names.astype(str).tolist()]
    exact_lookup = {name: name for name in exact_names}
    folded_lookup: dict[str, list[str]] = {}
    for name in exact_names:
        folded_lookup.setdefault(name.casefold(), []).append(name)
    return exact_lookup, folded_lookup


def _resolve_gene_name(adata, gene: str) -> str:
    raw = str(gene or "").strip()
    if not raw:
        return ""
    exact_lookup, folded_lookup = _build_gene_lookup(adata)
    if raw in exact_lookup:
        return exact_lookup[raw]
    matches = folded_lookup.get(raw.casefold(), [])
    if not matches:
        raise KeyError(f"Gene not found in adata.var_names: {raw}")
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Gene {raw!r} is ambiguous under case-insensitive matching: {', '.join(matches)}"
    )


def _resolve_gene_list(adata, genes: list[str]) -> tuple[list[str], list[str]]:
    valid_genes: list[str] = []
    missing_genes: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        try:
            canonical = _resolve_gene_name(adata, gene)
        except KeyError:
            missing_genes.append(str(gene))
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        valid_genes.append(canonical)
    return valid_genes, missing_genes


def _resolve_stage_map(adata, stage_key: str) -> list[str]:
    if stage_key not in adata.obs:
        return []
    series = adata.obs[stage_key]
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.astype(str).tolist())
        return [str(value) for value in series.dtype.categories.tolist() if str(value) in present]
    return [str(value) for value in pd.unique(series.astype(str))]


def _stage_display_label(stage: Any) -> str:
    text = str(stage).strip()
    if not text:
        return "Stage"
    return text if text.lower().startswith("stage") else f"Stage {text}"


def _is_baseline_stage_value(stage: Any) -> bool:
    text = str(stage).strip()
    if not text:
        return False
    folded = text.casefold()
    if folded in {"baseline", "stage 0"}:
        return True
    try:
        return float(text) == 0.0
    except ValueError:
        return False


def _read_stage_specific_perturbation_table(table_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(table_path)
    drop_columns = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
    return frame


def _resolve_stage_index(stages: list[str], stage_value: Any) -> int:
    normalized = str(stage_value).strip()
    if not normalized:
        raise ValueError("Stage value is required for stage-scoped perturbation rendering.")
    if normalized in stages:
        return stages.index(normalized)
    try:
        numeric_value = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unable to resolve stage index for {stage_value!r}.") from exc
    for index, stage in enumerate(stages):
        try:
            if float(str(stage).strip()) == numeric_value:
                return index
        except ValueError:
            continue
    raise ValueError(f"Unable to resolve stage index for {stage_value!r}. Available stages: {stages}")


def _stage_specific_track_table_path(analysis_dir: Path, *, mode: str, stage: str, track: str) -> Path:
    stage_dir = Path(analysis_dir) / "stage_specific_perturbation" / f"stage_{stage}"
    return stage_dir / f"{mode}_perturbation_track_{track}_stage_{stage}.csv"


def _normalize_stage_track_frame(frame: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    table = frame.copy()
    if "compound" in table.columns and mode == "pathway":
        table = table.rename(columns={"compound": "pathways"})
    if "drug_regulation" in table.columns:
        table = table.rename(columns={"drug_regulation": "regulated genes"})
    if "compound" in table.columns and mode == "single_gene":
        table = table.rename(columns={"compound": "gene"})
    if "compound" in table.columns and mode == "gene_combo":
        table = table.rename(columns={"compound": "gene_combination"})
    if mode == "pathway" and "perturbation score" in table.columns:
        table = table.sort_values("perturbation score", ascending=False)
    elif "pval_adjusted" in table.columns:
        table = table.sort_values("pval_adjusted", ascending=True)
    elif "perturbation score" in table.columns:
        table = table.sort_values("perturbation score", ascending=False)
    table = table.reset_index(drop=True)
    if "rank" not in table.columns:
        table.insert(0, "rank", np.arange(len(table)))
    return table


def _build_stage_track_options(adata, *, stages: list[str], track_values: Iterable[str]) -> list[dict[str, str]]:
    seen: set[str] = set()
    raw_values: list[str] = []
    include_overall = False
    for value in track_values:
        text = str(value).strip()
        if not text:
            continue
        if text == "overall":
            include_overall = True
            continue
        if text in seen:
            continue
        seen.add(text)
        raw_values.append(text)
    if not raw_values and not include_overall:
        return [{"value": "overall", "label": "Overall"}]
    raw_labels: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    for raw_value in raw_values:
        try:
            label = _track_display_label(adata, track=raw_value, stages=stages)
        except Exception:
            label = raw_value
        raw_labels[raw_value] = label
        label_counts[label] = label_counts.get(label, 0) + 1
    options = [{"value": "overall", "label": "Overall"}] if include_overall or not raw_values else []
    for raw_value in raw_values:
        label = raw_labels[raw_value]
        if label_counts[label] > 1:
            label = f"{label} ({raw_value})"
        options.append({"value": raw_value, "label": label})
    return options


def _normalize_stage_track_group_value(value: Any) -> str:
    text = str(value).strip()
    if not text or text == "overall":
        return "overall"
    try:
        return _track_cluster_id(text)
    except Exception:
        return text


def _first_perturb_change_for_mode(adata, *, mode: str) -> float:
    uns_key = PERTURBATION_MODE_KEYS[mode]
    payload = adata.uns.get(uns_key)
    if not isinstance(payload, dict) or not payload:
        raise KeyError(f"Missing {uns_key} in adata.uns")
    return float(_sorted_keys(payload.keys())[0])


def _stage_specific_track_group_table_path(analysis_dir: Path, *, mode: str, stage: str, track_group: str) -> Path:
    stage_dir = Path(analysis_dir) / "stage_specific_perturbation" / f"stage_{stage}"
    return stage_dir / f"{mode}_perturbation_track_group_{track_group}_stage_{stage}.csv"


def _generate_stage_specific_track_group_table(
    analysis_dir: Path,
    *,
    mode: str,
    stage: str,
    track_group: str,
) -> str:
    analysis_dir = Path(analysis_dir)
    adata, _dataset_path, _attribute_path = _read_analysis_adata(
        analysis_dir,
        backed=True,
        uns_keys=PERTURBATION_SCORING_UNS_KEYS,
    )
    try:
        stages = _resolve_stage_map(adata, "stage")
        stage_index = _resolve_stage_index(stages, stage)
        perturb_change = _first_perturb_change_for_mode(adata, mode=mode)
        scorer = perturbationAnalysis(
            adata,
            analysis_dir / "idrem",
            log2fc=perturb_change,
            stage=stage_index,
            mode=mode,
            direction_mode=None,
        )
        with redirect_stdout(io.StringIO()):
            results = scorer.main_analysis(
                perturbed_tracks=[str(track_group)],
                overall_perturbation_analysis=True,
                score="avg_backScore",
                items=None,
            )
        payload = results.get("overall")
        if not isinstance(payload, dict):
            raise KeyError(f"Unable to generate grouped stage-specific perturbation table for track group {track_group!r}.")
        table_payload = payload.get("top_compounds")
        if not isinstance(table_payload, dict):
            raise KeyError(f"Missing top_compounds for grouped stage-specific perturbation table {track_group!r}.")
        frame = _normalize_stage_track_frame(pd.DataFrame.from_dict(table_payload), mode=mode)
        table_path = _stage_specific_track_group_table_path(
            analysis_dir,
            mode=mode,
            stage=str(stage),
            track_group=str(track_group),
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(table_path, index=False)
        return str(table_path)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def _ensure_stage_specific_track_group_perturbation_table(
    analysis_dir: Path,
    *,
    mode: str,
    stage: str,
    track_group: str,
) -> str:
    if str(track_group) == "overall":
        raise ValueError("Overall stage-specific perturbation tables should be resolved from the precomputed stage table.")
    table_path = _stage_specific_track_group_table_path(
        analysis_dir,
        mode=mode,
        stage=str(stage),
        track_group=str(track_group),
    )
    if table_path.exists():
        return str(table_path)
    return _generate_stage_specific_track_group_table(
        analysis_dir,
        mode=mode,
        stage=str(stage),
        track_group=str(track_group),
    )


def _overall_track_group_table_path(
    analysis_dir: Path,
    *,
    mode: str,
    perturb_change: str,
    track_group: str,
) -> Path:
    root = Path(analysis_dir) / "overall_track_perturbation" / f"{mode}_change_{perturb_change}"
    return root / f"{mode}_perturbation_track_group_{track_group}.csv"


def _generate_overall_track_group_perturbation_tables(
    analysis_dir: Path,
    *,
    mode: str,
    perturb_change: str,
) -> dict[str, str]:
    analysis_dir = Path(analysis_dir)
    adata, _dataset_path, _attribute_path = _read_analysis_adata(
        analysis_dir,
        backed=False,
        uns_keys=PERTURBATION_SCORING_UNS_KEYS,
    )
    idrem_dir = analysis_dir / "idrem"
    if not idrem_dir.exists():
        raise FileNotFoundError(f"Missing idrem directory for perturbation analysis: {idrem_dir}")
    try:
        bound = float(str(perturb_change))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid perturb_change for overall track perturbation rendering: {perturb_change!r}") from exc
    scorer = perturbationAnalysis(
        adata,
        idrem_dir,
        stage=None,
        log2fc=bound,
        mode=mode,
        direction_mode=None,
    )
    delta_key = PERTURBATION_MODE_KEYS[mode].replace('_score', '_deltaD')
    delta_payload = adata.uns.get(delta_key, {})
    raw_tracks = (
        delta_payload.get(str(bound), {}).keys()
        if isinstance(delta_payload, dict) and isinstance(delta_payload.get(str(bound), {}), dict)
        else []
    )
    track_groups = []
    seen = set()
    for raw_track in _sorted_keys(raw_tracks):
        if str(raw_track) == "overall":
            continue
        group_value = _normalize_stage_track_group_value(raw_track)
        if group_value == "overall" or group_value in seen:
            continue
        seen.add(group_value)
        track_groups.append(group_value)
    if not track_groups:
        raise KeyError("Unable to generate grouped overall perturbation tables.")
    written: dict[str, str] = {}
    for track_group in track_groups:
        with io.StringIO() as buffer, redirect_stdout(buffer):
            results = scorer.main_analysis(
                perturbed_tracks=[str(track_group)],
                overall_perturbation_analysis=True,
                score='avg_backScore',
                items=None,
            )
        if not isinstance(results, dict):
            continue
        bucket = results.get("overall")
        if not isinstance(bucket, dict):
            continue
        table_payload = bucket.get("top_compounds") or bucket.get("down_compounds")
        if not isinstance(table_payload, dict):
            continue
        frame = _normalize_stage_track_frame(pd.DataFrame.from_dict(table_payload), mode=mode)
        table_path = _overall_track_group_table_path(
            analysis_dir,
            mode=mode,
            perturb_change=str(perturb_change),
            track_group=str(track_group),
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(table_path, index=False)
        written[str(track_group)] = str(table_path)
    return written


def _ensure_overall_track_group_perturbation_table(
    analysis_dir: Path,
    *,
    mode: str,
    perturb_change: str,
    track_group: str,
) -> str:
    table_path = _overall_track_group_table_path(
        analysis_dir,
        mode=mode,
        perturb_change=str(perturb_change),
        track_group=str(track_group),
    )
    if table_path.exists():
        return str(table_path)
    written = _generate_overall_track_group_perturbation_tables(
        analysis_dir,
        mode=mode,
        perturb_change=str(perturb_change),
    )
    resolved = written.get(str(track_group))
    if resolved:
        return resolved
    raise KeyError(
        f"Unable to generate grouped overall perturbation table for mode={mode!r}, "
        f"perturb_change={perturb_change!r}, track_group={track_group!r}."
    )


def _discover_overall_track_group_perturbation_tables(
    analysis_dir: Path,
    *,
    mode: str,
    perturb_change: str,
) -> dict[str, str]:
    root = Path(analysis_dir) / "overall_track_perturbation" / f"{mode}_change_{perturb_change}"
    if not root.exists():
        return {}
    prefix = f"{mode}_perturbation_track_group_"
    discovered: dict[str, str] = {}
    for path in sorted(root.glob(f"{mode}_perturbation_track_group_*.csv")):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        track_value = stem[len(prefix):].strip()
        if not track_value:
            continue
        discovered[track_value] = str(path)
    return discovered


def _discover_stage_specific_perturbations(analysis_dir: Path) -> dict[str, Any]:
    mode_prefixes = {
        "pathway": "pathway_perturbation_overall_stage_",
        "drug": "drug_perturbation_overall_stage_",
    }
    discovered_modes: dict[str, Any] = {}
    for mode, prefix in mode_prefixes.items():
        tables_by_stage: dict[str, str] = {}
        track_tables_by_stage: dict[str, dict[str, str]] = {}
        for path in sorted((Path(analysis_dir) / "stage_specific_perturbation").glob(f"stage_*/{prefix}*.csv")):
            stage_dir = path.parent.name
            if not stage_dir.startswith("stage_"):
                continue
            stage_value = stage_dir[len("stage_") :].strip()
            if not stage_value or _is_baseline_stage_value(stage_value):
                continue
            tables_by_stage[stage_value] = str(path)
        for path in sorted((Path(analysis_dir) / "stage_specific_perturbation").glob(f"stage_*/{mode}_perturbation_track_*_stage_*.csv")):
            stage_dir = path.parent.name
            if not stage_dir.startswith("stage_"):
                continue
            stage_value = stage_dir[len("stage_") :].strip()
            if not stage_value or _is_baseline_stage_value(stage_value):
                continue
            stem = path.stem
            prefix_text = f"{mode}_perturbation_track_"
            suffix_text = f"_stage_{stage_value}"
            if not stem.startswith(prefix_text) or not stem.endswith(suffix_text):
                continue
            track_value = stem[len(prefix_text) : -len(suffix_text)].strip()
            if not track_value:
                continue
            track_tables_by_stage.setdefault(stage_value, {})[track_value] = str(path)
        if tables_by_stage:
            ordered_stage_values = _sorted_keys(tables_by_stage.keys())
            discovered_modes[mode] = {
                "stage_options": [
                    {"value": stage_value, "label": _stage_display_label(stage_value)}
                    for stage_value in ordered_stage_values
                ],
                "tables_by_stage": tables_by_stage,
                "track_tables_by_stage": track_tables_by_stage,
            }
    return discovered_modes


def _dominant_text(series: pd.Series, *, field_name: str) -> str:
    cleaned = series.astype(str).replace({"nan": "", "None": ""})
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        raise ValueError(f"Expected non-empty values in adata.obs[{field_name!r}] for track labeling.")
    mode = cleaned.mode()
    if mode.empty:
        raise ValueError(f"Unable to resolve a dominant value from adata.obs[{field_name!r}] for track labeling.")
    return str(mode.iloc[0])


def _track_cluster_id(track: str) -> str:
    token = str(track).split("-")[0].strip()
    if not token or "n" in token:
        raise ValueError(
            f"Track {track!r} must start with exactly one stage-0 cluster id to build a display label."
        )
    return token


def _track_display_label(
    adata,
    *,
    track: str,
    stages: list[str],
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    celltype_key: str = "ident",
) -> str:
    if not stages:
        raise ValueError("Stage order is required to build progression-track labels.")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if cluster_key not in adata.obs:
        raise KeyError(f"Missing cluster key in adata.obs: {cluster_key}")
    if celltype_key not in adata.obs:
        raise KeyError(f"Missing cell type key in adata.obs: {celltype_key}")

    cluster_id = _track_cluster_id(str(track))
    first_stage = str(stages[0])
    stage_values = adata.obs[stage_key].astype(str)
    cluster_values = adata.obs[cluster_key].astype(str)
    mask = (stage_values == first_stage) & (cluster_values == cluster_id)
    if not bool(mask.any()):
        raise ValueError(
            f"Unable to resolve a first-stage cluster for track {track!r} at stage {first_stage!r}."
        )
    cell_type = _dominant_text(adata.obs.loc[mask, celltype_key], field_name=celltype_key)
    return f"track {cluster_id} {cell_type}"


def _umap_embedding_key(adata) -> str | None:
    if "X_umap" in adata.obsm:
        return "X_umap"
    if "umap" in adata.obsm:
        return "umap"
    return None


def _extract_matrix(adata, genes: list[str], *, layer: str | None = None) -> np.ndarray:
    working = adata.to_memory() if getattr(adata, "isbacked", False) and getattr(adata, "is_view", False) else adata
    matrix = working[:, genes].layers[layer] if layer else working[:, genes].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        array = array.reshape(working.n_obs, len(genes))
    return array


def _extract_gene_vector(adata, gene: str, *, layer: str | None = None) -> np.ndarray:
    matrix = _extract_matrix(adata, [gene], layer=layer)
    return matrix[:, 0].astype(float, copy=False)


def _plot_umap_snapshot(
    adata,
    *,
    stage: str,
    color_by: str,
    embedding_key: str,
    gene: str | None = None,
    stage_key: str = "stage",
):
    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in adata.obsm: {embedding_key}")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if gene in ("", None) and color_by not in adata.obs:
        raise KeyError(f"Missing color field in adata.obs: {color_by}")

    stage_mask = adata.obs[stage_key].astype(str).to_numpy() == str(stage)
    if not stage_mask.any():
        raise ValueError(f"No cells were found for stage={stage!r}.")

    coords = np.asarray(adata.obsm[embedding_key], dtype=float)[stage_mask]
    subset = adata[stage_mask]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=120)
    if gene not in ("", None):
        resolved_gene = _resolve_gene_name(subset, str(gene))
        values = _extract_gene_vector(subset, resolved_gene)
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=14,
            alpha=0.86,
            linewidths=0,
            c=values,
            cmap="viridis",
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(resolved_gene)
        title = f"UMAP snapshot: stage {stage} | gene {resolved_gene}"
    else:
        color_values = subset.obs[color_by].astype(str).replace({"nan": "", "None": ""})
        categories = [str(value) for value in pd.unique(color_values)]
        palette = build_palette(categories)
        for category in categories:
            category_mask = color_values.to_numpy() == category
            points = coords[category_mask]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=14,
                alpha=0.82,
                linewidths=0,
                color=palette[str(category)],
                label=str(category) or "(blank)",
            )
        if 0 < len(categories) <= 20:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
        else:
            ax.text(
                0.99,
                0.99,
                f"{len(categories)} groups",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="grey",
            )
        title = f"UMAP snapshot: stage {stage} | {color_by}"
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig, ax


def _track_mask(adata, *, track: str, stage_key: str, cluster_key: str, stages: list[str]) -> np.ndarray:
    if cluster_key not in adata.obs or stage_key not in adata.obs:
        return np.zeros(adata.n_obs, dtype=bool)
    stage_values = adata.obs[stage_key].astype(str).to_numpy()
    cluster_values = adata.obs[cluster_key].astype(str).to_numpy()
    allowed_pairs: set[tuple[str, str]] = set()
    for idx, token in enumerate(str(track).split("-")):
        if idx >= len(stages):
            break
        stage_label = str(stages[idx])
        for cluster in str(token).split("n"):
            cluster_id = cluster.strip()
            if cluster_id:
                allowed_pairs.add((stage_label, cluster_id))
    return np.fromiter(
        ((stage_values[i], cluster_values[i]) in allowed_pairs for i in range(adata.n_obs)),
        dtype=bool,
        count=adata.n_obs,
    )


def _custom_gene_table(
    adata,
    *,
    genes: list[str],
    scope: str,
    track: str | None,
    normalize_to_stage0: bool,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
) -> pd.DataFrame:
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if not genes:
        raise ValueError("At least one valid gene is required.")
    stages = _resolve_stage_map(adata, stage_key)
    if not stages:
        raise ValueError("No stages were found in adata.obs.")

    mask = np.ones(adata.n_obs, dtype=bool)
    if scope == "progression_track":
        if not track:
            raise ValueError("A progression track is required for progression_track scope.")
        mask = _track_mask(adata, track=track, stage_key=stage_key, cluster_key=cluster_key, stages=stages)
        if not mask.any():
            raise ValueError(f"Track {track!r} did not match any cells.")

    selected = adata[mask, :]
    matrix = _extract_matrix(selected, genes)
    stage_values = selected.obs[stage_key].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for gene_idx, gene in enumerate(genes):
        baseline: float | None = None
        for order, stage in enumerate(stages):
            stage_mask = stage_values == str(stage)
            if stage_mask.any():
                value = float(matrix[stage_mask, gene_idx].mean())
                n_cells = int(stage_mask.sum())
            else:
                value = float("nan")
                n_cells = 0
            if baseline is None and np.isfinite(value):
                baseline = value
            plotted = value
            if normalize_to_stage0 and np.isfinite(value):
                plotted = value - float(baseline or 0.0)
            rows.append(
                {
                    "gene": gene,
                    "stage": str(stage),
                    "stage_order": order,
                    "value": plotted,
                    "raw_value": value,
                    "n_cells": n_cells,
                    "scope": scope,
                    "track": track or "",
                }
            )
    return pd.DataFrame.from_records(rows)


def _plot_custom_gene_table(
    table: pd.DataFrame,
    *,
    normalize_to_stage0: bool,
    title: str,
):
    if table.empty:
        raise ValueError("No expression values were available for the selected genes.")
    stage_order = table["stage"].drop_duplicates().tolist()
    x = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.2, len(stage_order) * 1.5), 4.8), dpi=120)
    genes = table["gene"].drop_duplicates().tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(genes))))
    for idx, gene in enumerate(genes):
        gene_table = table[table["gene"] == gene].sort_values("stage_order")
        values = pd.to_numeric(gene_table["value"], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            label=gene,
            color=colors[idx % len(colors)],
        )
    if normalize_to_stage0:
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Expression change vs stage0")
    else:
        ax.set_ylabel("Mean expression")
    ax.set_xlabel("Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def _stage_dendrogram_plot(
    adata,
    *,
    stage: str,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    celltype_key: str = "ident",
):
    hcmarkers = adata.uns.get("hcmarkers")
    if not isinstance(hcmarkers, dict) or not hcmarkers:
        raise KeyError("Missing hcmarkers in adata.uns.")
    stage_bucket = hcmarkers.get(stage)
    if not isinstance(stage_bucket, dict) or "Z" not in stage_bucket:
        raise KeyError(f"Dendrogram data for stage={stage!r} is unavailable.")
    if stage_key not in adata.obs or cluster_key not in adata.obs:
        raise KeyError(f"Dendrogram requires adata.obs[{stage_key!r}] and adata.obs[{cluster_key!r}].")
    stage_adata = adata[adata.obs[stage_key].astype(str) == str(stage)]
    stage_adata.obs[cluster_key] = stage_adata.obs[cluster_key].astype(str)
    leaves = sch.leaves_list(stage_bucket["Z"])
    labels: list[str] = []
    for leaf in leaves:
        cluster = str(leaf)
        sub = stage_adata[stage_adata.obs[cluster_key] == cluster]
        label = cluster
        if sub.n_obs and celltype_key in sub.obs:
            label = f"{cluster} | {sub.obs[celltype_key].astype(str).mode().iloc[0]}"
        labels.append(label)
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.4), 4.2), dpi=120)
    sch.dendrogram(stage_bucket["Z"], ax=ax)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_title(f"Hierarchical dendrogram: stage {stage}")
    fig.tight_layout()
    return fig, ax


def _stage_dendrogram_plotly_spec(
    adata,
    *,
    stage: str,
    selected_cluster: str | None = None,
    node_color_lookup: Mapping[tuple[str, str], str] | None = None,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    celltype_key: str = "ident",
) -> dict[str, Any]:
    hcmarkers = adata.uns.get("hcmarkers")
    if not isinstance(hcmarkers, dict) or not hcmarkers:
        raise KeyError("Missing hcmarkers in adata.uns.")
    stage_bucket = hcmarkers.get(stage)
    if not isinstance(stage_bucket, dict) or "Z" not in stage_bucket:
        raise KeyError(f"Dendrogram data for stage={stage!r} is unavailable.")
    if stage_key not in adata.obs or cluster_key not in adata.obs:
        raise KeyError(f"Dendrogram requires adata.obs[{stage_key!r}] and adata.obs[{cluster_key!r}].")

    stage_mask = adata.obs[stage_key].astype(str).to_numpy() == str(stage)
    stage_obs = adata.obs.loc[stage_mask, [cluster_key] + ([celltype_key] if celltype_key in adata.obs else [])].copy()
    stage_obs[cluster_key] = stage_obs[cluster_key].astype(str)

    dendrogram = sch.dendrogram(stage_bucket["Z"], no_plot=True)
    leaf_positions = [5 + 10 * idx for idx in range(len(dendrogram["leaves"]))]
    leaf_clusters = [str(leaf) for leaf in dendrogram["leaves"]]
    leaf_labels: list[str] = []
    leaf_hover: list[str] = []
    selected_mask: list[bool] = []
    leaf_colors: list[str] = []
    for cluster in leaf_clusters:
        label = cluster
        ident = "NA"
        if celltype_key in stage_obs:
            subset = stage_obs.loc[stage_obs[cluster_key] == cluster, celltype_key]
            if not subset.empty:
                ident = str(subset.astype(str).mode().iloc[0])
                label = f"{cluster} | {ident}"
        leaf_labels.append(label)
        leaf_hover.append(f"Cluster: {cluster}<br>ident: {ident}")
        selected_mask.append(str(cluster) == str(selected_cluster))
        leaf_colors.append(
            str((node_color_lookup or {}).get((str(stage), str(cluster)), "#2D7D7C"))
        )

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for xs, ys in zip(dendrogram["icoord"], dendrogram["dcoord"]):
        line_x.extend([float(value) for value in xs] + [None])
        line_y.extend([float(value) for value in ys] + [None])

    selected_points = [idx for idx, active in enumerate(selected_mask) if active]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": line_x,
                "y": line_y,
                "hoverinfo": "skip",
                "line": {"color": "#6A7176", "width": 2},
                "showlegend": False,
            },
            {
                "type": "scatter",
                "mode": "markers",
                "x": leaf_positions,
                "y": [0.0] * len(leaf_positions),
                "text": leaf_hover,
                "customdata": leaf_clusters,
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": [13 if active else 9 for active in selected_mask],
                    "color": leaf_colors,
                    "opacity": 0.9,
                    "line": {
                        "width": [2 if active else 1 for active in selected_mask],
                        "color": ["#C15F31" if active else "#172124" for active in selected_mask],
                    },
                },
                "selectedpoints": selected_points,
                "showlegend": False,
            },
        ],
        "layout": {
            "height": max(280, min(440, 130 + 18 * len(leaf_labels))),
            "hovermode": "closest",
            "dragmode": "pan",
            "margin": {"l": 56, "r": 18, "t": 30, "b": 112},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "tickmode": "array",
                "tickvals": leaf_positions,
                "ticktext": leaf_labels,
                "tickangle": -45,
                "showgrid": False,
                "zeroline": False,
            },
            "yaxis": {
                "title": "Distance",
                "zeroline": False,
                "gridcolor": "rgba(31, 42, 46, 0.12)",
            },
            "title": {"text": f"Hierarchical dendrogram: stage {stage}", "x": 0.02, "xanchor": "left"},
        },
        "selected_cluster": str(selected_cluster) if selected_cluster not in (None, "") else "",
    }


def _stage_composition_plotly_spec(
    adata,
    *,
    stage_key: str = "stage",
    cell_type_key: str = "ident",
    min_label_fraction: float = 0.04,
) -> dict[str, Any]:
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if cell_type_key not in adata.obs:
        raise KeyError(f"Missing cell type key in adata.obs: {cell_type_key}")

    obs = adata.obs.loc[:, [stage_key, cell_type_key]].copy()
    obs[stage_key] = obs[stage_key].astype(str)
    obs[cell_type_key] = obs[cell_type_key].astype(str)

    stage_order = _resolve_stage_map(adata, stage_key)
    count_table = pd.crosstab(obs[stage_key], obs[cell_type_key]).reindex(stage_order, fill_value=0)
    fraction_table = (
        pd.crosstab(obs[stage_key], obs[cell_type_key], normalize="index")
        .reindex(stage_order, fill_value=0.0)
    )
    ordered_columns = sorted(fraction_table.columns.tolist())
    fraction_table = fraction_table.loc[:, ordered_columns]
    count_table = count_table.reindex(columns=ordered_columns, fill_value=0)
    palette_map = build_palette(ordered_columns)

    stage_labels = [str(stage) for stage in stage_order]
    traces: list[dict[str, Any]] = []
    for cell_type in ordered_columns:
        fractions = fraction_table[cell_type].astype(float).to_numpy()
        counts = count_table[cell_type].astype(int).to_numpy()
        text = [f"{value * 100:.1f}%" if value >= min_label_fraction else "" for value in fractions]
        customdata = [[int(count), f"{value * 100:.1f}%"] for count, value in zip(counts, fractions)]
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": str(cell_type),
                "y": stage_labels,
                "x": fractions.tolist(),
                "text": text,
                "textposition": "inside",
                "insidetextanchor": "middle",
                "marker": {"color": palette_map[str(cell_type)], "line": {"color": "white", "width": 0.6}},
                "customdata": customdata,
                "hovertemplate": (
                    "Cell type: %{fullData.name}<br>"
                    f"{stage_key}: "+"%{y}<br>"
                    "Fraction: %{customdata[1]}<br>"
                    "Cells: %{customdata[0]}<extra></extra>"
                ),
                "cliponaxis": False,
            }
        )

    return {
        "data": traces,
        "layout": {
            "height": max(360, min(560, 190 + 68 * len(stage_labels))),
            "barmode": "stack",
            "hovermode": "closest",
            "dragmode": "pan",
            "margin": {"l": 72, "r": 240, "t": 18, "b": 44},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "legend": {
                "orientation": "v",
                "x": 1.02,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "font": {"size": 11},
            },
            "xaxis": {
                "title": "Fraction",
                "range": [0, 1],
                "tickformat": ".0%",
                "gridcolor": "rgba(31, 42, 46, 0.12)",
                "zeroline": False,
            },
            "yaxis": {
                "title": stage_key,
                "autorange": "reversed",
            },
        },
        "stage_order": stage_labels,
        "cell_type_order": [str(value) for value in ordered_columns],
    }


def _branch_heatmap_plotly_spec(
    adata,
    *,
    stage: str,
    cluster: str,
    level: str,
    direction: str,
    top_n: int,
) -> dict[str, Any]:
    heatmap_fig, _ax, matrix, _markers = plot_hierarchical_marker_branch_heatmap(
        adata,
        stage=stage,
        cluster=cluster,
        level=level,
        direction=direction,
        top_n=top_n,
        show=False,
    )
    plt.close(heatmap_fig)
    return {
        "data": [
            {
                "type": "heatmap",
                "x": [str(value) for value in matrix.columns.tolist()],
                "y": [str(value) for value in matrix.index.tolist()],
                "z": matrix.to_numpy(dtype=float).tolist(),
                "colorscale": "RdBu",
                "zmid": 0,
                "hovertemplate": "Gene: %{y}<br>Cluster: %{x}<br>Score: %{z:.4f}<extra></extra>",
                "colorbar": {"title": "z-score"},
            }
        ],
        "layout": {
            "height": max(420, min(980, 160 + 22 * len(matrix.index))),
            "margin": {"l": 110, "r": 24, "t": 42, "b": 92},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"tickangle": -35, "side": "bottom"},
            "yaxis": {"autorange": "reversed"},
            "title": {
                "text": (
                    f"Hierarchical markers: stage={stage}, cluster={cluster}, "
                    f"level={level}, direction={HIERARCHICAL_DIRECTION_LABELS[direction]}"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
        },
    }


def _extract_metadata(adata, analysis_dir: Path | None = None) -> dict[str, Any]:
    stages = _resolve_stage_map(adata, "stage")
    umap_embedding_key = _umap_embedding_key(adata)
    hierarchical_stages: list[dict[str, Any]] = []
    hcmarkers = adata.uns.get("hcmarkers")
    if isinstance(hcmarkers, dict):
        ordered_hcmarker_stages = [stage for stage in stages if stage in hcmarkers]
        ordered_hcmarker_stages.extend(str(stage) for stage in hcmarkers.keys() if str(stage) not in ordered_hcmarker_stages)
        for stage in ordered_hcmarker_stages:
            stage_bucket = hcmarkers.get(stage) or {}
            markers = stage_bucket.get("markers") if isinstance(stage_bucket, dict) else None
            if not isinstance(markers, dict):
                continue
            clusters: list[dict[str, Any]] = []
            for cluster in _sorted_keys(markers.keys()):
                levels = markers.get(cluster)
                if not isinstance(levels, dict):
                    continue
                level_values = _sorted_keys(levels.keys())
                if level_values:
                    clusters.append({"value": str(cluster), "levels": level_values})
            if clusters:
                hierarchical_stages.append({"value": str(stage), "clusters": clusters})

    progression = adata.uns.get("progressionMarkers")
    dynamic_tracks: list[dict[str, Any]] = []
    if isinstance(progression, dict):
        for track in _sorted_keys(progression.keys()):
            bucket = progression.get(track) or {}
            if isinstance(bucket, dict):
                dynamic_tracks.append(
                    {
                        "value": str(track),
                        "label": _track_display_label(adata, track=str(track), stages=stages),
                        "directions": _ordered_dynamic_directions(bucket.keys()),
                    }
                )

    perturbation_modes: dict[str, Any] = {}
    for mode, uns_key in PERTURBATION_MODE_KEYS.items():
        payload = adata.uns.get(uns_key)
        if not isinstance(payload, dict):
            continue
        changes: dict[str, list[str]] = {}
        for perturb_change in _sorted_keys(payload.keys()):
            track_bucket = payload.get(perturb_change)
            if not isinstance(track_bucket, dict):
                continue
            ordered_tracks = [track for track in _sorted_keys(track_bucket.keys()) if str(track) != "overall"]
            if "overall" in track_bucket:
                ordered_tracks = ["overall"] + ordered_tracks
            tracks = _build_stage_track_options(
                adata,
                stages=stages,
                track_values=ordered_tracks,
            )
            if tracks:
                changes[str(perturb_change)] = tracks
        if changes:
            perturbation_modes[mode] = {
                "perturb_changes": list(changes.keys()),
                "tracks_by_change": changes,
            }
    stage_specific_modes = _discover_stage_specific_perturbations(Path(analysis_dir)) if analysis_dir else {}

    var_names = [str(name) for name in adata.var_names.astype(str).tolist()]
    default_genes = var_names[: min(3, len(var_names))]
    umap_color_options: list[str] = []
    umap_selection_fields: list[str] = []
    if umap_embedding_key:
        preferred_color_fields = ["stage", "ident", "leiden", "name.simple", "final.ident", "Sample.ID"]
        seen_fields: set[str] = set()
        for field in preferred_color_fields + [str(column) for column in adata.obs.columns]:
            if field in seen_fields or field not in adata.obs:
                continue
            seen_fields.add(field)
            series = adata.obs[field]
            if series.isna().all():
                continue
            unique_count = int(series.astype(str).nunique())
            if unique_count <= 1:
                continue
            if unique_count <= 80 or field in {"stage", "ident", "leiden"}:
                umap_color_options.append(field)
        for field in ["barcode", "stage", "leiden", "ident", "name.simple", "final.ident", "Sample.ID"]:
            if field in adata.obs and field not in umap_selection_fields:
                umap_selection_fields.append(field)

    dynamic_panel_available = bool(dynamic_tracks or stages)
    available_modes = [
        mode
        for mode in ("umap", "hierarchical", "dynamic_markers", "perturbation")
        if (
            (mode == "umap" and bool(umap_embedding_key))
            or (mode == "hierarchical" and hierarchical_stages)
            or (mode == "dynamic_markers" and dynamic_panel_available)
            or (mode == "perturbation" and (perturbation_modes or stage_specific_modes))
        )
    ]
    default_mode = next(
        (mode for mode in ("dynamic_markers", "hierarchical", "perturbation", "umap") if mode in available_modes),
        available_modes[0] if available_modes else "dynamic_markers",
    )
    return {
        "umap": {
            "available": bool(umap_embedding_key),
            "embedding_key": umap_embedding_key,
            "stages": stages,
            "color_options": umap_color_options,
            "default_color_by": next((field for field in ("ident", "leiden", "stage") if field in umap_color_options), umap_color_options[0] if umap_color_options else ""),
            "selection_fields": umap_selection_fields,
            "reason": None if umap_embedding_key else "No UMAP embedding is available. Expected adata.obsm['X_umap'] or adata.obsm['umap'].",
        },
        "hierarchical": {
            "available": bool(hierarchical_stages),
            "stages": hierarchical_stages,
            "reason": None if hierarchical_stages else "No hierarchical marker data is available. This usually means the analysis ran with ignore_hcmarkers=true.",
        },
        "dynamic_markers": {
            "available": dynamic_panel_available,
            "progression_available": bool(dynamic_tracks),
            "tracks": dynamic_tracks,
            "reason": None if dynamic_panel_available else "No progression marker data or stage metadata is available for dynamic marker exploration.",
        },
        "custom_genes": {
            "available": bool(stages),
            "stages": stages,
            "tracks": dynamic_tracks,
            "reason": None if stages else "Stage metadata is unavailable for custom gene exploration.",
            "gene_limit": GENE_LIMIT,
            "gene_count": int(adata.n_vars),
            "default_genes": default_genes,
        },
        "perturbation": {
            "available": bool(perturbation_modes or stage_specific_modes),
            "modes": perturbation_modes,
            "stage_specific_modes": stage_specific_modes,
            "reason": None if (perturbation_modes or stage_specific_modes) else "No perturbation outputs are available. This usually means perturbation analysis was disabled or the relevant mode was skipped.",
        },
        "default_mode": default_mode,
    }


@dataclass
class AnalysisRecord:
    analysis_dir: Path
    dataset_path: Path
    attribute_path: Path | None
    adata: Any
    metadata: dict[str, Any]
    full: bool


class AnalysisExplorerService:
    def __init__(self, *, skill_root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], AnalysisRecord] = {}
        self._temporal_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._stage_composition_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.umap_cache = UmapViewportCache(skill_root=skill_root)
        self.idrem_cache = IdremTrackSummaryCache()

    def _analysis_identity(self, analysis_dir: Path) -> tuple[Any, ...]:
        analysis_dir = Path(analysis_dir).expanduser().resolve()
        dataset_path = analysis_dir / "dataset.h5ad"
        attribute_path = analysis_dir / "attribute.pkl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Analysis dataset not found: {dataset_path}")
        return (
            str(analysis_dir),
            _cache_key(dataset_path),
            _cache_key(attribute_path if attribute_path.exists() else None),
            _stage_specific_perturbation_identity(analysis_dir),
        )

    def _load(self, analysis_dir: Path, *, full: bool = False) -> AnalysisRecord:
        analysis_dir = Path(analysis_dir).expanduser().resolve()
        key = self._analysis_identity(analysis_dir) + (bool(full),)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        adata, resolved_dataset, resolved_attribute = _read_analysis_adata(
            analysis_dir,
            backed=not full,
            uns_keys=INTERACTIVE_UNS_KEYS,
        )
        record = AnalysisRecord(
            analysis_dir=analysis_dir,
            dataset_path=resolved_dataset,
            attribute_path=resolved_attribute,
            adata=adata,
            metadata=_extract_metadata(adata, analysis_dir=analysis_dir),
            full=bool(full),
        )
        with self._lock:
            self._cache[key] = record
        return record

    def get_temporal_graph_payload(self, analysis_dir: str | Path) -> dict[str, Any]:
        analysis_path = Path(analysis_dir)
        key = self._analysis_identity(analysis_path)
        with self._lock:
            cached = self._temporal_graph_cache.get(key)
            if cached is not None:
                return cached

        record = self._load(analysis_path, full=False)
        if 'edges' not in record.adata.uns:
            fallback_edges = _load_temporal_edges_fallback(analysis_path)
            if fallback_edges is not None:
                record.adata.uns['edges'] = fallback_edges
        fig, _ax, nodes, edges = plot_temporal_dynamics_graph(record.adata, show=False)
        plt.close(fig)

        node_lookup = {
            row["node"]: row
            for _, row in nodes.iterrows()
        }
        stage_order = _resolve_stage_map(record.adata, "stage")
        stage_tick_values: list[float] = []
        stage_tick_labels: list[str] = []
        for stage in stage_order:
            stage_nodes = nodes.loc[nodes["stage"].astype(str) == str(stage)]
            if stage_nodes.empty:
                continue
            stage_tick_values.append(float(stage_nodes["y"].astype(float).mean()))
            stage_text = str(stage)
            stage_tick_labels.append(stage_text if stage_text.lower().startswith("stage") else f"Stage {stage_text}")

        payload_nodes: list[dict[str, Any]] = []
        for index, row in nodes.reset_index(drop=True).iterrows():
            payload_node = {
                "index": int(index),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "stage": str(row["stage"]),
                "cluster": str(row["cluster"]),
                "cell_type": str(row.get("cell_type", "NA")),
                "n_cells": int(round(float(row.get("n_cells", 0.0)))),
                "node_size": float(row.get("node_size", 0.0)),
                "color": str(row.get("node_color", "#6E7C80")),
            }
            if "track_label" in row and str(row["track_label"]).strip():
                payload_node["track_label"] = str(row["track_label"])
            if "track_key" in row and str(row["track_key"]).strip():
                payload_node["track_key"] = str(row["track_key"])
            payload_nodes.append(payload_node)

        payload_edges: list[dict[str, Any]] = []
        for _, row in edges.iterrows():
            src = node_lookup.get(row["src_node"])
            dst = node_lookup.get(row["dst_node"])
            if src is None or dst is None:
                continue
            payload_edge = {
                "x0": float(src["x"]),
                "y0": float(src["y"]),
                "x1": float(dst["x"]),
                "y1": float(dst["y"]),
                "src_stage": str(row.get("src_stage", src["stage"])),
                "src_cluster": str(row.get("src_cluster", src["cluster"])),
                "dst_stage": str(row.get("dst_stage", dst["stage"])),
                "dst_cluster": str(row.get("dst_cluster", dst["cluster"])),
            }
            if "track_label" in row and str(row["track_label"]).strip():
                payload_edge["track_label"] = str(row["track_label"])
            payload_edges.append(payload_edge)

        payload = {
            "node_count": len(payload_nodes),
            "edge_count": len(payload_edges),
            "stage_order": stage_order,
            "stage_tick_values": stage_tick_values,
            "stage_tick_labels": stage_tick_labels,
            "nodes": payload_nodes,
            "edges": payload_edges,
        }
        with self._lock:
            self._temporal_graph_cache[key] = payload
        return payload

    def get_stage_composition_payload(self, analysis_dir: str | Path) -> dict[str, Any]:
        analysis_path = Path(analysis_dir)
        key = self._analysis_identity(analysis_path)
        with self._lock:
            cached = self._stage_composition_cache.get(key)
            if cached is not None:
                return cached

        record = self._load(analysis_path, full=False)
        payload = _stage_composition_plotly_spec(record.adata)
        with self._lock:
            self._stage_composition_cache[key] = payload
        return payload

    def get_hierarchical_dendrogram_spec(self, analysis_dir: str | Path, *, stage: str, cluster: str | None = None) -> dict[str, Any]:
        analysis_path = Path(analysis_dir)
        record = self._load(analysis_path, full=False)
        temporal_payload = self.get_temporal_graph_payload(analysis_path)
        node_color_lookup = {
            (str(node.get("stage", "")), str(node.get("cluster", ""))): str(node.get("color", "#2D7D7C"))
            for node in temporal_payload.get("nodes", [])
        }
        return _stage_dendrogram_plotly_spec(
            record.adata,
            stage=str(stage),
            selected_cluster=str(cluster or ""),
            node_color_lookup=node_color_lookup,
        )

    def get_hierarchical_heatmap_spec(
        self,
        analysis_dir: str | Path,
        *,
        stage: str,
        cluster: str,
        level: str,
        direction: str,
        top_n: int,
    ) -> dict[str, Any]:
        record = self._load(Path(analysis_dir), full=False)
        return _branch_heatmap_plotly_spec(
            record.adata,
            stage=str(stage),
            cluster=str(cluster),
            level=str(level),
            direction=str(direction),
            top_n=int(top_n),
        )

    def get_idrem_track_summary(self, analysis_dir: str | Path, *, track_key: str, track_label: str = "") -> dict[str, Any]:
        return self.idrem_cache.get_summary(analysis_dir, track_key=str(track_key), track_label=str(track_label or ""))

    def build_view(self, analysis_dir: str | Path, form_data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        analysis_path = Path(analysis_dir)
        record = self._load(analysis_path, full=False)
        metadata = record.metadata
        umap_enabled = _bool_value(form_data.get("enable_umap") if form_data else None, default=False)
        requested_mode = str(form_data.get("mode")) if form_data and form_data.get("mode") is not None else None
        visible_modes = [
            value
            for value, _label in EXPLORER_MODES
            if not (value == "umap" and not umap_enabled)
        ]
        active_mode = "dynamic_markers" if requested_mode == "custom_genes" else _resolve_string(
            visible_modes,
            requested_mode,
            default=metadata.get("default_mode") or "dynamic_markers",
        )
        umap = self._resolve_umap_state(record.adata, metadata, form_data)
        hierarchical = self._resolve_hierarchical_state(metadata, form_data)
        dynamic_markers = self._resolve_dynamic_state(record.adata, metadata, form_data)
        perturbation = self._resolve_perturbation_state(analysis_path, record.adata, metadata, form_data)
        if active_mode == "dynamic_markers" and dynamic_markers.get("view_mode") == "custom_genes":
            record = self._load(analysis_path, full=True)
            metadata = record.metadata
            umap = self._resolve_umap_state(record.adata, metadata, form_data)
            hierarchical = self._resolve_hierarchical_state(metadata, form_data)
            dynamic_markers = self._resolve_dynamic_state(record.adata, metadata, form_data)
            perturbation = self._resolve_perturbation_state(analysis_path, record.adata, metadata, form_data)
        result = self._render_result(record, active_mode, umap, hierarchical, dynamic_markers, perturbation)
        return {
            "analysis_dir": str(record.analysis_dir),
            "active_mode": active_mode,
            "mode_options": [
                {
                    "value": value,
                    "label": label,
                    "available": bool(metadata.get(value, {}).get("available")),
                }
                for value, label in EXPLORER_MODES
                if value != "umap" or umap_enabled
            ],
            "metadata": {
                "gene_count": metadata["custom_genes"]["gene_count"],
                "gene_limit": metadata["custom_genes"]["gene_limit"],
                "dataset_path": str(record.dataset_path),
                "attribute_path": str(record.attribute_path) if record.attribute_path else None,
                "umap_available": bool(metadata["umap"]["available"]),
                "umap_enabled": umap_enabled,
            },
            "umap": umap,
            "hierarchical": hierarchical,
            "dynamic_markers": dynamic_markers,
            "perturbation": perturbation,
            "result": result,
        }

    def get_umap_payload(
        self,
        analysis_dir: str | Path,
        *,
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
    ) -> dict[str, Any]:
        return self.umap_cache.get_payload(
            analysis_dir,
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

    def summarize_umap_selection(
        self,
        analysis_dir: str | Path,
        *,
        stage: str,
        color_by: str,
        cell_ids: list[int],
        gene: str | None = None,
        render_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.umap_cache.summarize_selection(
            analysis_dir,
            stage=stage,
            color_by=color_by,
            cell_ids=cell_ids,
            gene=gene,
            render_mode=render_mode,
        )

    def export_publication_pdf(
        self,
        analysis_dir: str | Path,
        *,
        form_data: Mapping[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        record = self._load(Path(analysis_dir), full=True)
        metadata = record.metadata
        requested_mode = str(form_data.get("mode")) if form_data and form_data.get("mode") is not None else None
        active_mode = "dynamic_markers" if requested_mode == "custom_genes" else _resolve_string(
            [mode for mode, _label in EXPLORER_MODES],
            requested_mode,
            default=metadata.get("default_mode") or "dynamic_markers",
        )
        figure_key = str(form_data.get("figure_key") if form_data else "").strip()
        umap = self._resolve_umap_state(record.adata, metadata, form_data)
        hierarchical = self._resolve_hierarchical_state(metadata, form_data)
        dynamic_markers = self._resolve_dynamic_state(record.adata, metadata, form_data)
        perturbation = self._resolve_perturbation_state(Path(analysis_dir), record.adata, metadata, form_data)

        if active_mode == "umap":
            if not umap["available"]:
                raise KeyError(umap["reason"])
            fig, _ax = _plot_umap_snapshot(
                record.adata,
                stage=umap["stage"],
                color_by=umap["color_by"],
                gene=umap["gene"],
                embedding_key=str(metadata["umap"]["embedding_key"]),
            )
            suffix = f"gene_{umap['gene']}" if umap["gene"] else umap["color_by"]
            return _figure_to_pdf_bytes(fig), f"umap_stage_{umap['stage']}_{suffix}.pdf"

        if active_mode == "hierarchical":
            if figure_key not in {"dendrogram", "branch_heatmap"}:
                raise ValueError("figure_key is required for hierarchical explorer PDF export.")
            if not hierarchical["available"]:
                raise KeyError(hierarchical["reason"])
            if figure_key == "dendrogram":
                fig, _ax = _stage_dendrogram_plot(record.adata, stage=hierarchical["stage"])
                name = f"hierarchical_dendrogram_stage_{hierarchical['stage']}.pdf"
            else:
                fig, _ax, _matrix, _markers = plot_hierarchical_marker_branch_heatmap(
                    record.adata,
                    stage=hierarchical["stage"],
                    cluster=hierarchical["cluster"],
                    level=hierarchical["level"],
                    direction=hierarchical["direction"],
                    top_n=hierarchical["top_n"],
                    show=False,
                )
                name = (
                    f"hierarchical_heatmap_stage_{hierarchical['stage']}"
                    f"_cluster_{hierarchical['cluster']}_level_{hierarchical['level']}.pdf"
                )
            return _figure_to_pdf_bytes(fig), name

        if active_mode == "dynamic_markers":
            if dynamic_markers["view_mode"] == "custom_genes":
                fig, _table, _warnings = self._build_custom_gene_content(record.adata, dynamic_markers["genes"])
                return _figure_to_pdf_bytes(fig), "custom_gene_trajectories.pdf"
            fig, _table, _subtitle = self._build_dynamic_content(record.adata, dynamic_markers["ranked"])
            return _figure_to_pdf_bytes(fig), f"dynamic_markers_{dynamic_markers['ranked']['plot_kind']}.pdf"

        if active_mode == "perturbation":
            fig, _table, _work = self._build_perturbation_content(record.adata, perturbation)
            return _figure_to_pdf_bytes(fig), f"{perturbation['mode']}_perturbation_ranked_bar.pdf"

        raise ValueError(f"Unsupported explorer mode for PDF export: {active_mode}")

    def export_table_csv(
        self,
        analysis_dir: str | Path,
        *,
        form_data: Mapping[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        record = self._load(Path(analysis_dir), full=False)
        metadata = record.metadata
        requested_mode = str(form_data.get("mode")) if form_data and form_data.get("mode") is not None else None
        active_mode = "dynamic_markers" if requested_mode == "custom_genes" else _resolve_string(
            [mode for mode, _label in EXPLORER_MODES],
            requested_mode,
            default=metadata.get("default_mode") or "dynamic_markers",
        )
        if active_mode == "hierarchical":
            hierarchical = self._resolve_hierarchical_state(metadata, form_data)
            if not hierarchical["available"]:
                raise KeyError(hierarchical["reason"])
            table = _hierarchical_marker_export_frame(self._extract_hierarchical_marker_table(record.adata, hierarchical))
            csv_bytes = table.to_csv(index=False).encode("utf-8")
            filename = (
                f"hierarchical_markers_stage_{hierarchical['stage']}"
                f"_cluster_{hierarchical['cluster']}_level_{hierarchical['level']}_{hierarchical['direction']}.csv"
            )
            return csv_bytes, filename

        if active_mode == "dynamic_markers":
            dynamic_markers = self._resolve_dynamic_state(record.adata, metadata, form_data)
            if not dynamic_markers["available"]:
                raise KeyError(dynamic_markers["reason"])
            if dynamic_markers["view_mode"] == "custom_genes":
                custom = dynamic_markers["genes"]
                if len(custom["genes"]) > custom["gene_limit"]:
                    raise ValueError(f"Please limit requests to at most {custom['gene_limit']} genes.")
                if not custom["valid_genes"]:
                    raise ValueError("None of the requested genes were found in adata.var_names.")
                table = _custom_gene_table(
                    record.adata,
                    genes=custom["valid_genes"],
                    scope=custom["scope"],
                    track=custom["track"] if custom["scope"] == "progression_track" else None,
                    normalize_to_stage0=custom["normalize_to_stage0"],
                )
                filename = f"custom_genes_{custom['scope']}.csv"
                return table.to_csv(index=False).encode("utf-8"), filename

            ranked = dynamic_markers["ranked"]
            if not ranked["available"]:
                raise KeyError(ranked["reason"])
            fig, table, _subtitle = self._build_dynamic_content(record.adata, ranked)
            plt.close(fig)
            filename = f"dynamic_markers_{ranked['plot_kind']}_{ranked['track']}.csv"
            return table.to_csv(index=False).encode("utf-8"), filename

        if active_mode == "perturbation":
            perturbation = self._resolve_perturbation_state(Path(analysis_dir), record.adata, metadata, form_data)
            if not perturbation["available"]:
                raise KeyError(perturbation["reason"])
            _fig, table, _work = self._build_perturbation_content(record.adata, perturbation)
            scope_suffix = (
                f"stage_{perturbation['stage']}"
                if perturbation["scope"] == "stage"
                else f"{perturbation['track']}_change_{perturbation['perturb_change']}"
            )
            filename = f"{perturbation['mode']}_perturbation_{scope_suffix}.csv"
            return table.to_csv(index=False).encode("utf-8"), filename

        raise ValueError("CSV table export is currently available only for hierarchical markers, dynamic markers, and perturbation.")

    def _build_hierarchical_content(self, adata, state: dict[str, Any]):
        dendrogram_fig, _ax = _stage_dendrogram_plot(adata, stage=state["stage"])
        heatmap_fig, _ax2, _matrix, _markers = plot_hierarchical_marker_branch_heatmap(
            adata,
            stage=state["stage"],
            cluster=state["cluster"],
            level=state["level"],
            direction=state["direction"],
            top_n=state["top_n"],
            show=False,
        )
        marker_table = self._extract_hierarchical_marker_table(adata, state)
        return dendrogram_fig, heatmap_fig, marker_table

    def _extract_hierarchical_marker_table(self, adata, state: dict[str, Any]) -> pd.DataFrame:
        return extract_hierarchical_marker_table(
            adata,
            stage=state["stage"],
            cluster=state["cluster"],
            level=state["level"],
            direction=state["direction"],
            top_n=state["top_n"],
        )

    def _build_dynamic_content(self, adata, state: dict[str, Any]):
        base_kwargs = {
            "tracks": [state["track"]],
            "directions": tuple(state["directions"]),
            "top_n": state["top_n"],
            "normalize_to_stage0": state["normalize_to_stage0"],
            "show": False,
        }
        if state["plot_kind"] == "heatmap":
            fig, _axes, table = plot_dynamic_marker_heatmap(
                adata,
                per_track_total=state["per_track_total"],
                **base_kwargs,
            )
            subtitle = "track heatmap"
        elif state["plot_kind"] == "cell_heatmap":
            fig, _axes, table = plot_dynamic_marker_heatmap(
                adata,
                cell_level=True,
                per_track_total=state["per_track_total"],
                **base_kwargs,
            )
            subtitle = "cell-level heatmap"
        else:
            fig, _axes, table = plot_dynamic_marker_trajectories(adata, **base_kwargs)
            subtitle = "trajectory plot"
        return fig, table, subtitle

    def _build_custom_gene_content(self, adata, state: dict[str, Any]):
        warnings: list[str] = []
        if len(state["genes"]) > state["gene_limit"]:
            raise ValueError(f"Please limit requests to at most {state['gene_limit']} genes.")
        if state["missing_genes"]:
            warnings.append(f"Ignored missing genes: {', '.join(state['missing_genes'])}")
        if not state["valid_genes"]:
            raise ValueError("None of the requested genes were found in adata.var_names.")
        table = _custom_gene_table(
            adata,
            genes=state["valid_genes"],
            scope=state["scope"],
            track=state["track"] if state["scope"] == "progression_track" else None,
            normalize_to_stage0=state["normalize_to_stage0"],
        )
        fig, _ax = _plot_custom_gene_table(
            table,
            normalize_to_stage0=state["normalize_to_stage0"],
            title="Custom gene trajectories"
            if state["scope"] == "global_stage"
            else f"Custom gene trajectories: {state['track_label']}",
        )
        return fig, table, warnings

    def _build_perturbation_content(self, adata, state: dict[str, Any]):
        if state.get("table_path"):
            table = _read_stage_specific_perturbation_table(state["table_path"])
        else:
            table = extract_perturbation_table(
                adata,
                mode=state["mode"],
                perturb_change=state["perturb_change"],
                track=state["track"],
            )
        if state["top_n"] > 0:
            table = table.head(state["top_n"]).copy()
        use_pval = state["mode"] != "pathway"
        fig, _ax, work = plot_perturbation_ranked_bar(table, top_n=state["top_n"], use_neglog10_pval=use_pval, show=False)
        return fig, table, work

    def _resolve_umap_state(self, adata, metadata: dict[str, Any], form_data: Mapping[str, Any] | None) -> dict[str, Any]:
        section = metadata["umap"]
        if not section["available"]:
            return {"available": False, "reason": section["reason"], "color_options": [], "color_by": ""}
        raw_gene = form_data.get("umap_gene") if form_data else None
        resolved_gene = ""
        if raw_gene not in (None, ""):
            resolved_gene = _resolve_gene_name(adata, str(raw_gene))
        return {
            "available": True,
            "reason": None,
            "stage": _resolve_string(
                section.get("stages", []),
                form_data.get("umap_stage") if form_data else None,
                default=section.get("stages", [""])[0] if section.get("stages") else "",
            ),
            "stage_options": section.get("stages", []),
            "color_options": section["color_options"],
            "color_by": _resolve_string(
                section["color_options"],
                form_data.get("umap_color_by") if form_data else None,
                default=section["default_color_by"],
            ),
            "gene": resolved_gene,
            "selection_fields": section["selection_fields"],
        }

    def _resolve_hierarchical_state(self, metadata: dict[str, Any], form_data: Mapping[str, Any] | None) -> dict[str, Any]:
        section = metadata["hierarchical"]
        if not section["available"]:
            return {"available": False, "reason": section["reason"]}
        stage_options = section["stages"]
        stage = _resolve_string(
            [item["value"] for item in stage_options],
            form_data.get("hier_stage") if form_data else None,
            default=stage_options[0]["value"],
        )
        stage_entry = next(item for item in stage_options if item["value"] == stage)
        cluster = _resolve_string(
            [item["value"] for item in stage_entry["clusters"]],
            form_data.get("hier_cluster") if form_data else None,
            default=stage_entry["clusters"][0]["value"],
        )
        cluster_entry = next(item for item in stage_entry["clusters"] if item["value"] == cluster)
        level = _resolve_string(
            cluster_entry["levels"],
            form_data.get("hier_level") if form_data else None,
            default=cluster_entry["levels"][0],
        )
        direction = "up"
        return {
            "available": True,
            "reason": None,
            "stage": stage,
            "cluster": cluster,
            "level": level,
            "direction": direction,
            "direction_label": HIERARCHICAL_DIRECTION_LABELS[direction],
            "top_n": _int_value(form_data.get("hier_top_n") if form_data else None, default=20, minimum=1, maximum=100),
            "stage_options": stage_options,
            "cluster_options": stage_entry["clusters"],
            "level_options": cluster_entry["levels"],
        }

    def _resolve_dynamic_state(self, adata, metadata: dict[str, Any], form_data: Mapping[str, Any] | None) -> dict[str, Any]:
        section = metadata["dynamic_markers"]
        custom_state = self._resolve_custom_gene_state(adata, metadata, form_data)
        if not section["available"] and not custom_state["available"]:
            return {"available": False, "reason": section["reason"]}

        ranked_available = bool(section["progression_available"])
        if ranked_available:
            track_options = section["tracks"]
            track = _resolve_string(
                [item["value"] for item in track_options],
                form_data.get("dyn_track") if form_data else None,
                default=track_options[0]["value"],
            )
            track_entry = next(item for item in track_options if item["value"] == track)
            direction_options = track_entry["directions"] or ["increasing"]
            directions = [item for item in _form_list(form_data, "dyn_directions") if item in direction_options]
            if not directions:
                directions = ["increasing"] if "increasing" in direction_options else [direction_options[0]]
            ranked_state = {
                "available": True,
                "reason": None,
                "track": track,
                "track_label": track_entry["label"],
                "track_options": track_options,
                "direction_options": direction_options,
                "directions": directions,
                "plot_kind": _resolve_string(
                    ("trajectories", "heatmap", "cell_heatmap"),
                    form_data.get("dyn_plot_kind") if form_data else None,
                    default="trajectories",
                ),
                "top_n": _int_value(form_data.get("dyn_top_n") if form_data else None, default=8, minimum=1, maximum=50),
                "per_track_total": _bool_value(form_data.get("dyn_per_track_total") if form_data else None, default=False),
                "normalize_to_stage0": _bool_value(form_data.get("dyn_normalize_to_stage0") if form_data else None, default=True),
            }
        else:
            ranked_state = {
                "available": False,
                "reason": "No progression marker data is available. This usually means the analysis ran with ignore_dynamic_markers=true.",
                "track_options": [],
                "direction_options": [],
                "directions": [],
                "plot_kind": "trajectories",
                "top_n": 8,
                "per_track_total": False,
                "normalize_to_stage0": True,
            }

        view_mode_options: list[dict[str, str]] = []
        if ranked_state["available"]:
            view_mode_options.append({"value": "ranked_markers", "label": "Dynamic Markers"})
        if custom_state["available"]:
            view_mode_options.append({"value": "custom_genes", "label": "Custom Genes"})
        requested_mode = str(form_data.get("mode")) if form_data and form_data.get("mode") is not None else ""
        default_view_mode = "custom_genes" if requested_mode == "custom_genes" else ("ranked_markers" if ranked_state["available"] else "custom_genes")
        return {
            "available": bool(view_mode_options),
            "reason": None,
            "view_mode": _resolve_string(
                [item["value"] for item in view_mode_options],
                form_data.get("dyn_view_mode") if form_data else None,
                default=default_view_mode,
            ),
            "view_mode_options": view_mode_options,
            "ranked": ranked_state,
            "genes": custom_state,
        }

    def _resolve_custom_gene_state(self, adata, metadata: dict[str, Any], form_data: Mapping[str, Any] | None) -> dict[str, Any]:
        section = metadata["custom_genes"]
        if not section["available"]:
            return {"available": False, "reason": section["reason"], "gene_text": "", "missing_genes": []}
        track_options = section["tracks"]
        default_gene_text = ", ".join(section["default_genes"])
        raw_gene_text = form_data.get("gene_list") if form_data else default_gene_text
        gene_text = str(raw_gene_text or "").strip() or default_gene_text
        parsed_genes = _split_gene_text(gene_text)
        valid_genes, missing_genes = _resolve_gene_list(adata, parsed_genes)
        track = _resolve_string(
            [item["value"] for item in track_options],
            form_data.get("gene_track") if form_data else None,
            default=track_options[0]["value"] if track_options else "",
        )
        track_entry = next((item for item in track_options if item["value"] == track), None)
        scope = _resolve_string(("global_stage", "progression_track"), form_data.get("gene_scope") if form_data else None, default="global_stage")
        return {
            "available": True,
            "reason": None,
            "scope": scope,
            "scope_label": CUSTOM_GENE_SCOPE_LABELS[scope],
            "track": track,
            "track_label": track_entry["label"] if track_entry else "",
            "track_options": track_options,
            "show_track": scope == "progression_track" and bool(track_options),
            "gene_text": gene_text,
            "genes": parsed_genes,
            "valid_genes": valid_genes,
            "missing_genes": missing_genes,
            "normalize_to_stage0": _bool_value(form_data.get("gene_normalize_to_stage0") if form_data else None, default=True),
            "gene_limit": section["gene_limit"],
        }

    def _resolve_perturbation_state(
        self,
        analysis_dir: Path,
        adata,
        metadata: dict[str, Any],
        form_data: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        section = metadata["perturbation"]
        if not section["available"]:
            return {"available": False, "reason": section["reason"]}
        stage_specific_modes = section.get("stage_specific_modes") or {}
        standard_modes = section.get("modes") or {}
        scope_options: list[str] = []
        if standard_modes:
            scope_options.append("overall")
        if stage_specific_modes:
            scope_options.append("stage")
        scope = _resolve_string(
            scope_options,
            form_data.get("pert_scope") if form_data else None,
            default="stage" if stage_specific_modes and not standard_modes else "overall",
        )
        top_n = _int_value(form_data.get("pert_top_n") if form_data else None, default=20, minimum=1, maximum=100)

        if scope == "stage":
            mode_options = list(stage_specific_modes.keys())
            if not mode_options:
                scope = "overall"
            else:
                mode = _resolve_string(
                    mode_options,
                    form_data.get("pert_mode") if form_data else None,
                    default=mode_options[0],
                )
                mode_entry = stage_specific_modes[mode]
                stage_options = mode_entry["stage_options"]
                stage = _resolve_string(
                    [item["value"] for item in stage_options],
                    form_data.get("pert_stage") if form_data else None,
                    default=stage_options[0]["value"],
                )
                stage_entry = next(item for item in stage_options if item["value"] == stage)
                representative_tracks: dict[str, str] = {}
                for item in metadata.get("dynamic_markers", {}).get("tracks", []):
                    raw_value = str(item["value"])
                    representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)
                standard_mode_entry = (standard_modes.get(mode) or {})
                for track_list in (standard_mode_entry.get("tracks_by_change") or {}).values():
                    for item in track_list:
                        raw_value = str(item["value"])
                        if raw_value == "overall":
                            continue
                        representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)
                for raw_value in _sorted_keys(
                    (mode_entry.get("track_tables_by_stage", {}) or {}).get(stage, {}).keys()
                ):
                    representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)
                track_options = [{"value": "overall", "label": "Overall"}]
                for group_value in _sorted_keys(key for key in representative_tracks.keys() if key != "overall"):
                    representative_track = representative_tracks[group_value]
                    try:
                        label = _track_display_label(
                            adata,
                            track=representative_track,
                            stages=metadata["umap"]["stages"],
                        )
                    except Exception:
                        label = f"track {group_value}"
                    track_options.append({"value": group_value, "label": label})
                requested_track = _normalize_stage_track_group_value(
                    form_data.get("pert_track") if form_data else None
                )
                track = _resolve_string(
                    [item["value"] for item in track_options],
                    requested_track,
                    default="overall",
                )
                if track == "overall":
                    table_path = mode_entry["tables_by_stage"][stage]
                else:
                    table_path = _ensure_stage_specific_track_group_perturbation_table(
                        Path(analysis_dir),
                        mode=mode,
                        stage=stage,
                        track_group=track,
                    )
                return {
                    "available": True,
                    "reason": None,
                    "scope": "stage",
                    "scope_options": scope_options,
                    "mode": mode,
                    "mode_options": mode_options,
                    "stage": stage,
                    "stage_label": stage_entry["label"],
                    "stage_options": stage_options,
                    "table_path": table_path,
                    "perturb_change": None,
                    "perturb_change_options": [],
                    "track": track,
                    "track_label": next((item["label"] for item in track_options if item["value"] == track), track),
                    "track_options": track_options,
                    "top_n": top_n,
                }

        mode_options = list(standard_modes.keys())
        mode = _resolve_string(
            mode_options,
            form_data.get("pert_mode") if form_data else None,
            default=mode_options[0],
        )
        mode_entry = standard_modes[mode]
        perturb_change = _resolve_string(
            mode_entry["perturb_changes"],
            form_data.get("perturb_change") if form_data else None,
            default=mode_entry["perturb_changes"][0],
        )
        available_track_options = mode_entry["tracks_by_change"][perturb_change]
        raw_non_overall_values = [
            str(item["value"])
            for item in available_track_options
            if str(item["value"]) != "overall"
        ]
        normalized_non_overall_values = [
            _normalize_stage_track_group_value(value)
            for value in raw_non_overall_values
        ]
        precomputed_group_values = set()
        if raw_non_overall_values and (
            len(set(raw_non_overall_values)) == len(set(normalized_non_overall_values))
            and set(raw_non_overall_values) == set(normalized_non_overall_values)
        ):
            precomputed_group_values = set(raw_non_overall_values)
        generated_track_tables = _discover_overall_track_group_perturbation_tables(
            Path(analysis_dir),
            mode=mode,
            perturb_change=perturb_change,
        )
        if not precomputed_group_values and not generated_track_tables:
            try:
                generated_track_tables = _generate_overall_track_group_perturbation_tables(
                    Path(analysis_dir),
                    mode=mode,
                    perturb_change=perturb_change,
                )
            except Exception:
                generated_track_tables = {}
        track_values = ["overall"]
        if precomputed_group_values:
            track_values.extend(_sorted_keys(precomputed_group_values))
        else:
            track_values.extend(_sorted_keys(generated_track_tables.keys()))
        track_options = _build_stage_track_options(
            adata,
            stages=metadata["umap"]["stages"],
            track_values=track_values,
        )
        requested_track = _normalize_stage_track_group_value(
            form_data.get("pert_track") if form_data else None
        )
        track = _resolve_string(
            [item["value"] for item in track_options],
            requested_track,
            default="overall" if any(item["value"] == "overall" for item in track_options) else track_options[0]["value"],
        )
        track_entry = next(item for item in track_options if item["value"] == track)
        table_path = ""
        if track != "overall" and track not in precomputed_group_values:
            table_path = generated_track_tables.get(str(track), "")
            if not table_path:
                table_path = _ensure_overall_track_group_perturbation_table(
                    Path(analysis_dir),
                    mode=mode,
                    perturb_change=perturb_change,
                    track_group=track,
                )
        return {
            "available": True,
            "reason": None,
            "scope": "overall",
            "scope_options": scope_options,
            "mode": mode,
            "mode_options": mode_options,
            "perturb_change": perturb_change,
            "perturb_change_options": mode_entry["perturb_changes"],
            "track": track,
            "track_label": track_entry["label"],
            "track_options": track_options,
            "stage": "",
            "stage_label": "",
            "stage_options": [],
            "table_path": table_path,
            "top_n": top_n,
        }

    def _render_result(
        self,
        record: AnalysisRecord,
        active_mode: str,
        umap: dict[str, Any],
        hierarchical: dict[str, Any],
        dynamic_markers: dict[str, Any],
        perturbation: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            if active_mode == "umap":
                return self._render_umap(umap)
            if active_mode == "hierarchical":
                return self._render_hierarchical(record.adata, hierarchical)
            if active_mode == "dynamic_markers":
                return self._render_dynamic(record.adata, dynamic_markers)
            if active_mode == "perturbation":
                return self._render_perturbation(record.adata, perturbation)
            return {"status": "error", "title": "Unavailable", "message": f"Unsupported explorer mode: {active_mode}"}
        except Exception as exc:
            return {
                "status": "error",
                "title": "Explorer render failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def _render_umap(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state["available"]:
            return {"status": "unavailable", "title": "UMAP", "message": state["reason"]}
        return {
            "status": "ok",
            "title": "Interactive UMAP Explorer",
            "subtitle": (
                f"stage={state['stage']} · color_by=gene ({state['gene']}) · use box/lasso select or click a cell"
                if state["gene"]
                else f"stage={state['stage']} · color_by={state['color_by']} · use box/lasso select or click a cell"
            ),
            "plots": [],
            "split_cards": [],
            "tables": [],
            "warnings": [],
            "umap_widget": {
                "stage": state["stage"],
                "color_options": state["color_options"],
                "default_color_by": state["color_by"],
                "color_mode": "gene" if state["gene"] else "metadata",
                "gene": state["gene"],
                "selection_fields": state["selection_fields"],
                "publication_query": {
                    "mode": "umap",
                    "figure_key": "umap",
                    "umap_stage": state["stage"],
                    "umap_color_by": state["color_by"],
                    "umap_gene": state["gene"],
                },
            },
        }

    def _render_hierarchical(self, adata, state: dict[str, Any]) -> dict[str, Any]:
        if not state["available"]:
            return {"status": "unavailable", "title": "Hierarchical Markers", "message": state["reason"]}
        cluster_levels = {
            str(item["value"]): [str(level) for level in item["levels"]]
            for item in state["cluster_options"]
        }
        return {
            "status": "ok",
            "title": "Hierarchical Marker Explorer",
            "subtitle": f"stage={state['stage']} cluster={state['cluster']} level={state['level']}",
            "plots": [],
            "split_cards": [],
            "tables": [],
            "warnings": [],
            "hierarchical_widget": {
                "stage": state["stage"],
                "cluster": state["cluster"],
                "level": state["level"],
                "direction": state["direction"],
                "direction_label": state["direction_label"],
                "top_n": state["top_n"],
                "cluster_levels": cluster_levels,
                "dendrogram_query": {
                    "stage": state["stage"],
                    "cluster": state["cluster"],
                },
                "heatmap_query": {
                    "stage": state["stage"],
                    "cluster": state["cluster"],
                    "level": state["level"],
                    "direction": state["direction"],
                    "top_n": state["top_n"],
                },
                "dendrogram_publication_query": {
                    "mode": "hierarchical",
                    "figure_key": "dendrogram",
                    "hier_stage": state["stage"],
                    "hier_cluster": state["cluster"],
                    "hier_level": state["level"],
                    "hier_direction": state["direction"],
                    "hier_top_n": state["top_n"],
                },
                "heatmap_publication_query": {
                    "mode": "hierarchical",
                    "figure_key": "branch_heatmap",
                    "hier_stage": state["stage"],
                    "hier_cluster": state["cluster"],
                    "hier_level": state["level"],
                    "hier_direction": state["direction"],
                    "hier_top_n": state["top_n"],
                },
                "table_download_query": {
                    "mode": "hierarchical",
                    "hier_stage": state["stage"],
                    "hier_cluster": state["cluster"],
                    "hier_level": state["level"],
                    "hier_direction": state["direction"],
                    "hier_top_n": state["top_n"],
                },
            },
        }

    def _render_dynamic(self, adata, state: dict[str, Any]) -> dict[str, Any]:
        if not state["available"]:
            return {"status": "unavailable", "title": "Dynamic Markers", "message": state["reason"]}
        if state["view_mode"] == "custom_genes":
            custom = state["genes"]
            if not custom["available"]:
                return {"status": "unavailable", "title": "Dynamic Markers", "message": custom["reason"]}
            try:
                fig, table, warnings = self._build_custom_gene_content(adata, custom)
            except ValueError as exc:
                return {
                    "status": "error",
                    "title": "Dynamic Markers",
                    "message": str(exc),
                }
            subtitle = (
                f"view=custom genes | scope={custom['scope_label']} | {custom['track_label']} | genes={', '.join(custom['valid_genes'])}"
                if custom["show_track"]
                else f"view=custom genes | scope={custom['scope_label']} | genes={', '.join(custom['valid_genes'])}"
            )
            return {
                "status": "ok",
                "title": "Dynamic Marker Explorer",
                "subtitle": subtitle,
                "plots": [
                    {
                        "label": "Trajectory plot",
                        "image_data": _figure_to_data_uri(fig),
                        "publication_query": {
                            "mode": "dynamic_markers",
                            "figure_key": "trajectory",
                            "dyn_view_mode": "custom_genes",
                            "gene_scope": custom["scope"],
                            "gene_track": custom["track"],
                            "gene_list": custom["gene_text"],
                            "gene_normalize_to_stage0": "true" if custom["normalize_to_stage0"] else "false",
                        },
                        "table_download_query": {
                            "mode": "dynamic_markers",
                            "dyn_view_mode": "custom_genes",
                            "gene_scope": custom["scope"],
                            "gene_track": custom["track"],
                            "gene_list": custom["gene_text"],
                            "gene_normalize_to_stage0": "true" if custom["normalize_to_stage0"] else "false",
                        },
                    }
                ],
                "tables": [],
                "warnings": warnings,
            }

        ranked = state["ranked"]
        if not ranked["available"]:
            return {"status": "unavailable", "title": "Dynamic Markers", "message": ranked["reason"]}
        fig, table, subtitle = self._build_dynamic_content(adata, ranked)
        return {
            "status": "ok",
            "title": "Dynamic Marker Explorer",
            "subtitle": f"view=dynamic markers | {ranked['track_label']} | plot={subtitle} | directions={', '.join(ranked['directions'])}",
            "plots": [
                {
                    "label": "Dynamic marker plot",
                    "image_data": _figure_to_data_uri(fig),
                    "publication_query": {
                        "mode": "dynamic_markers",
                        "figure_key": ranked["plot_kind"],
                        "dyn_view_mode": "ranked_markers",
                        "dyn_track": ranked["track"],
                        "dyn_directions": ranked["directions"],
                        "dyn_plot_kind": ranked["plot_kind"],
                        "dyn_top_n": ranked["top_n"],
                        "dyn_per_track_total": "true" if ranked["per_track_total"] else "false",
                        "dyn_normalize_to_stage0": "true" if ranked["normalize_to_stage0"] else "false",
                    },
                    "table_download_query": {
                        "mode": "dynamic_markers",
                        "dyn_view_mode": "ranked_markers",
                        "dyn_track": ranked["track"],
                        "dyn_directions": ranked["directions"],
                        "dyn_plot_kind": ranked["plot_kind"],
                        "dyn_top_n": ranked["top_n"],
                        "dyn_per_track_total": "true" if ranked["per_track_total"] else "false",
                        "dyn_normalize_to_stage0": "true" if ranked["normalize_to_stage0"] else "false",
                    },
                }
            ],
            "tables": [],
            "warnings": [],
        }

    def _render_perturbation(self, adata, state: dict[str, Any]) -> dict[str, Any]:
        if not state["available"]:
            return {"status": "unavailable", "title": "Perturbation", "message": state["reason"]}
        fig, table, _work = self._build_perturbation_content(adata, state)
        if state["scope"] == "stage":
            if state["track"] == "overall":
                track_text = state["stage_label"]
                title = f"{state['mode']} perturbation | {track_text}"
                subtitle = f"scope=stage | mode={state['mode']} | {track_text}"
            else:
                track_text = f"{state['stage_label']} | {state['track_label']}"
                title = f"{state['mode']} perturbation | {track_text}"
                subtitle = f"scope=stage | mode={state['mode']} | {state['stage_label']} | {state['track_label']}"
        else:
            track_text = state["track_label"] if state["track"] != "overall" else "Overall"
            title = f"{state['mode']} perturbation | perturb_change={state['perturb_change']} | {track_text}"
            subtitle = f"scope=overall | mode={state['mode']} | perturb_change={state['perturb_change']} | {track_text}"
        return {
            "status": "ok",
            "title": "Perturbation Explorer",
            "subtitle": subtitle,
            "plots": [
                {
                    "label": "Ranked bar plot",
                    "image_data": _figure_to_data_uri(fig),
                    "plotly_spec": _perturbation_plotly_spec(table, title=title),
                    "publication_query": {
                        "mode": "perturbation",
                        "figure_key": "ranked_bar",
                        "pert_scope": state["scope"],
                        "pert_mode": state["mode"],
                        "perturb_change": state["perturb_change"],
                        "pert_track": state["track"],
                        "pert_stage": state["stage"],
                        "pert_top_n": state["top_n"],
                    },
                    "table_download_query": {
                        "mode": "perturbation",
                        "pert_scope": state["scope"],
                        "pert_mode": state["mode"],
                        "perturb_change": state["perturb_change"],
                        "pert_track": state["track"],
                        "pert_stage": state["stage"],
                        "pert_top_n": state["top_n"],
                    },
                }
            ],
            "tables": [],
            "warnings": [],
        }
