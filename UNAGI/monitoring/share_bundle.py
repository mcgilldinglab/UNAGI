from __future__ import annotations

import ast
import html
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from UNAGI.plotting import (
    extract_dynamic_marker_table,
    extract_hierarchical_marker_table,
    extract_perturbation_table,
    plot_temporal_dynamics_graph,
    plot_perturbation_ranked_bar,
    plot_stage_umap_grid,
)

from .figure_export import export_static_publication_pdf
from .interactive_analysis import (
    PERTURBATION_MODE_KEYS,
    _build_stage_track_options,
    _discover_stage_specific_perturbations,
    _discover_overall_track_group_perturbation_tables,
    _ensure_overall_track_group_perturbation_table,
    _ensure_stage_specific_track_group_perturbation_table,
    _extract_metadata,
    _generate_overall_track_group_perturbation_tables,
    _normalize_stage_track_group_value,
)
from .idrem_summary import IdremTrackSummaryCache
from .visualization_pipeline import (
    INTERACTIVE_UNS_KEYS,
    _file_fingerprint,
    _read_analysis_adata,
    generate_analysis_visualizations,
)

SHARE_BUNDLE_DIR_NAME = "share_bundle"
SHARE_BUNDLE_VERSION = 5
OFFLINE_MARKER_TOP_N = 20
OFFLINE_PERTURBATION_TOP_N = 20
SHARE_BUNDLE_PUBLICATION_EXPORTS = {
    "stage_composition",
    "temporal_dynamics_graph",
    "pathway_ranked_bar",
    "drug_ranked_bar",
    "single_gene_ranked_bar",
    "gene_combo_ranked_bar",
}

ARTIFACT_TITLE_MAP = {
    "stage_umap_grid": "Stage UMAP Grid",
    "stage_composition": "Stage Composition",
    "temporal_dynamics_graph": "Temporal Dynamics Graph",
    "pathway_perturbation_table": "Pathway Perturbation Table",
    "pathway_ranked_bar": "Pathway Ranked Bar",
    "drug_perturbation_table": "Drug Perturbation Table",
    "drug_ranked_bar": "Drug Ranked Bar",
    "single_gene_perturbation_table": "Single Gene Perturbation Table",
    "single_gene_ranked_bar": "Single Gene Ranked Bar",
    "gene_combo_perturbation_table": "Gene Combination Perturbation Table",
    "gene_combo_ranked_bar": "Gene Combination Ranked Bar",
    "dynamic_marker_table": "Dynamic Marker Table",
    "dynamic_marker_trajectories": "Dynamic Marker Trajectories",
    "dynamic_marker_heatmap": "Dynamic Marker Heatmap",
    "hierarchical_marker_heatmap": "Hierarchical Marker Heatmap",
    "hc_dendrograms": "Hierarchical Dendrograms",
}

OVERVIEW_ENTRY_NAMES = {"stage_umap_grid", "stage_composition"}
TEMPORAL_ENTRY_NAMES = {"temporal_dynamics_graph"}
PERTURBATION_ENTRY_NAMES = {
    "pathway_perturbation_table",
    "pathway_ranked_bar",
    "drug_perturbation_table",
    "drug_ranked_bar",
    "single_gene_perturbation_table",
    "single_gene_ranked_bar",
    "gene_combo_perturbation_table",
    "gene_combo_ranked_bar",
}
DYNAMIC_ENTRY_NAMES = {
    "dynamic_marker_table",
    "dynamic_marker_trajectories",
    "dynamic_marker_heatmap",
}
HIERARCHICAL_ENTRY_NAMES = {
    "hierarchical_marker_heatmap",
    "hc_dendrograms",
}


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix == ".json":
        return "json"
    return "file"


def _humanize_entry_name(name: str) -> str:
    if name in ARTIFACT_TITLE_MAP:
        return ARTIFACT_TITLE_MAP[name]
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else str(name)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned or "artifact"


def _bundle_download_stem(prefix: str, *parts: Any) -> str:
    tokens: list[str] = []
    prefix_token = _safe_name(prefix)
    if prefix_token and prefix_token != "artifact":
        tokens.append(prefix_token)
    for part in parts:
        token = _safe_name(str(part or ""))
        if token and token != "artifact":
            tokens.append(token)
    return "_".join(tokens) or "artifact"


def _bundle_paths(analysis_dir: Path) -> tuple[Path, Path, Path]:
    bundle_dir = analysis_dir / SHARE_BUNDLE_DIR_NAME
    manifest_path = bundle_dir / "manifest.json"
    zip_path = analysis_dir / f"{SHARE_BUNDLE_DIR_NAME}.zip"
    return bundle_dir, manifest_path, zip_path


def _existing_bundle_manifest(analysis_dir: Path) -> dict[str, Any] | None:
    bundle_dir, manifest_path, _legacy_zip_path = _bundle_paths(analysis_dir)
    if not bundle_dir.exists() or not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    zip_path = Path(str(payload.get("zip_path") or ""))
    if not zip_path.exists():
        return None
    return payload


def _share_bundle_base_name(analysis_path: Path, analysis_label: str | None, generated_at: datetime | None = None) -> str:
    label = str(analysis_label or analysis_path.parent.name or analysis_path.name).strip()
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"unagi_offline_report_{_safe_name(label)}_{stamp}"


def _cleanup_old_share_bundle_archives(analysis_dir: Path) -> None:
    for candidate in analysis_dir.glob("unagi_offline_report_*.zip"):
        try:
            candidate.unlink()
        except OSError:
            pass
    legacy = analysis_dir / f"{SHARE_BUNDLE_DIR_NAME}.zip"
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass


def _source_fingerprint(analysis_dir: Path) -> dict[str, Any]:
    return {
        "version": SHARE_BUNDLE_VERSION,
        "dataset": _file_fingerprint(analysis_dir / "dataset.h5ad"),
        "attribute": _file_fingerprint(analysis_dir / "attribute.pkl"),
        "visualizations_manifest": _file_fingerprint(analysis_dir / "visualizations" / "manifest.json"),
        "idrem_summary_index": _file_fingerprint(analysis_dir / "idrem_track_summaries" / "index.json"),
    }


def _analysis_iteration_from_dir(analysis_dir: Path) -> str | None:
    for parent in analysis_dir.parents:
        name = str(parent.name)
        if name.startswith("iter") and name[4:].isdigit():
            return name[4:]
    return None


def _load_temporal_edges_fallback(analysis_dir: Path) -> Mapping[str, Any] | None:
    iteration = _analysis_iteration_from_dir(analysis_dir)
    if not iteration:
        return None
    try:
        training_root = analysis_dir.parents[3]
    except IndexError:
        return None
    edges_path = training_root / "input" / iteration / "edges.txt"
    if not edges_path.exists():
        return None
    try:
        return json.loads(json.dumps(ast.literal_eval(edges_path.read_text(encoding="utf-8"))))
    except Exception:
        return None


def _copy_artifact(src: Path, dest_dir: Path, used_names: set[str]) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(src.stem)
    suffix = src.suffix.lower()
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    dest_path = dest_dir / candidate
    shutil.copy2(src, dest_path)
    return str(dest_path)


def _reserve_unique_artifact_path(dest_path: Path, used_names: set[str]) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(dest_path.stem)
    suffix = dest_path.suffix.lower()
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return dest_path.parent / candidate


def _csv_preview(path: Path, *, rows: int = 16) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    preview = frame.head(rows).fillna("")
    return {
        "columns": [str(column) for column in preview.columns.tolist()],
        "rows": preview.astype(str).values.tolist(),
        "row_count": int(len(frame)),
    }


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_generated_csv(frame: pd.DataFrame, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination.name


def _sanitize_excel_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[:\/?*\[\]]+", "_", str(name or "Sheet1")).strip().strip("'")
    if not cleaned:
        cleaned = "Sheet1"
    return cleaned[:31]


def _excel_column_name(index: int) -> str:
    value = index + 1
    result = ""
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _excel_inline_text(value: Any) -> str:
    return f'<is><t xml:space="preserve">{xml_escape("" if value is None else str(value))}</t></is>'


def _xlsx_cell_xml(row_index: int, col_index: int, value: Any) -> str:
    cell_ref = f"{_excel_column_name(col_index)}{row_index}"
    if value is None:
        return f'<c r="{cell_ref}" t="inlineStr">{_excel_inline_text("")}</c>'
    if isinstance(value, pd.Timestamp):
        return f'<c r="{cell_ref}" t="inlineStr">{_excel_inline_text(value.isoformat())}</c>'
    if isinstance(value, (np.bool_, bool)):
        return f'<c r="{cell_ref}" t="b"><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{int(value)}</v></c>'
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isfinite(numeric):
            return f'<c r="{cell_ref}"><v>{numeric}</v></c>'
        return f'<c r="{cell_ref}" t="inlineStr">{_excel_inline_text("")}</c>'
    try:
        if pd.isna(value):
            return f'<c r="{cell_ref}" t="inlineStr">{_excel_inline_text("")}</c>'
    except Exception:
        pass
    return f'<c r="{cell_ref}" t="inlineStr">{_excel_inline_text(value)}</c>'


def _write_generated_xlsx(frame: pd.DataFrame, destination: Path, *, sheet_name: str = "Sheet1") -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_sheet_name = _sanitize_excel_sheet_name(sheet_name)
    normalized = frame.copy().where(pd.notna(frame), None)

    rows_xml: list[str] = []
    header_cells = "".join(
        _xlsx_cell_xml(1, column_index, column_name)
        for column_index, column_name in enumerate(normalized.columns.tolist())
    )
    rows_xml.append(f'<row r="1">{header_cells}</row>')
    for row_offset, row_values in enumerate(normalized.itertuples(index=False, name=None), start=2):
        cells = "".join(
            _xlsx_cell_xml(row_offset, column_index, value)
            for column_index, value in enumerate(row_values)
        )
        rows_xml.append(f'<row r="{row_offset}">{cells}</row>')

    last_row = max(1, len(normalized) + 1)
    last_col = max(1, len(normalized.columns))
    dimension = f"A1:{_excel_column_name(last_col - 1)}{last_row}"
    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(rows_xml)}</sheetData>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f'<sheet name="{xml_escape(safe_sheet_name)}" sheetId="1" r:id="rId1"/>'
        '</sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", root_rels_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook.writestr("xl/styles.xml", styles_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
    return destination.name


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    payload = json.loads(frame.replace({np.nan: None}).to_json(orient="records"))
    return payload if isinstance(payload, list) else []


def _frame_table_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in frame.columns.tolist()],
        "rows": frame.fillna("").astype(str).values.tolist(),
        "row_count": int(len(frame)),
        "records": _frame_records(frame),
    }


def _sorted_keys(values: Iterable[Any]) -> list[str]:
    def sort_key(value: Any) -> tuple[int, Any]:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(values, key=sort_key)]


def _build_dynamic_marker_bundle(
    adata,
    *,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, Any]:
    metadata = _extract_metadata(adata)
    section = metadata.get("dynamic_markers") or {}
    tracks = list(section.get("tracks") or [])
    if not tracks:
        return {"available": False, "tracks": [], "tables": {}}

    tables_dir = bundle_dir / "assets" / "marker_tables" / "dynamic"
    table_payloads: dict[str, Any] = {}
    serialized_tracks: list[dict[str, Any]] = []
    for track in tracks:
        track_value = str(track.get("value") or "")
        track_label = str(track.get("label") or track_value)
        directions = [str(value) for value in track.get("directions") or []]
        available_directions: list[str] = []
        for direction in directions:
            frame = extract_dynamic_marker_table(
                adata,
                track=track_value,
                direction=direction,
                top_n=OFFLINE_MARKER_TOP_N,
                sort_by="rank",
            )
            if frame.empty:
                continue
            stage_columns = [str(column) for column in frame.columns if str(column).startswith("stage")]
            key = f"{track_value}::{direction}"
            filename = _bundle_download_stem(
                download_prefix,
                "dynamic_markers",
                track_label or track_value,
                f"direction_{direction}",
            ) + ".csv"
            csv_path = tables_dir / filename
            xlsx_path = csv_path.with_suffix(".xlsx")
            _write_generated_csv(frame, csv_path)
            _write_generated_xlsx(frame, xlsx_path, sheet_name=f"dynamic_{track_value}_{direction}")
            table_payloads[key] = {
                "track": track_value,
                "track_label": track_label,
                "direction": direction,
                "stage_columns": stage_columns,
                "table": _frame_table_payload(frame),
                "download_path": _relpath(xlsx_path, bundle_dir),
                "download_label": "Download Excel",
                "csv_download_path": _relpath(csv_path, bundle_dir),
            }
            available_directions.append(direction)
        if available_directions:
            serialized_tracks.append(
                {
                    "value": track_value,
                    "label": track_label,
                    "directions": available_directions,
                }
            )

    return {
        "available": bool(serialized_tracks),
        "tracks": serialized_tracks,
        "tables": table_payloads,
    }


def _build_hierarchical_marker_bundle(
    adata,
    *,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, Any]:
    metadata = _extract_metadata(adata)
    section = metadata.get("hierarchical") or {}
    stages = list(section.get("stages") or [])
    if not stages:
        return {"available": False, "stages": [], "tables": {}}

    tables_dir = bundle_dir / "assets" / "marker_tables" / "hierarchical"
    table_payloads: dict[str, Any] = {}
    serialized_stages: list[dict[str, Any]] = []
    for stage in stages:
        stage_value = str(stage.get("value") or "")
        clusters_out: list[dict[str, Any]] = []
        for cluster in stage.get("clusters") or []:
            cluster_value = str(cluster.get("value") or "")
            levels = [str(level) for level in cluster.get("levels") or []]
            available_levels: list[str] = []
            directions_by_level: dict[str, list[str]] = {}
            for level in levels:
                level_directions: list[str] = []
                for direction in ("up", "down"):
                    frame = extract_hierarchical_marker_table(
                        adata,
                        stage=stage_value,
                        cluster=cluster_value,
                        level=level,
                        direction=direction,
                        top_n=OFFLINE_MARKER_TOP_N,
                    )
                    if frame.empty:
                        continue
                    filename = _bundle_download_stem(
                        download_prefix,
                        "hierarchical_markers",
                        f"stage_{stage_value}",
                        f"cluster_{cluster_value}",
                        f"level_{level}",
                        f"direction_{direction}",
                    ) + ".csv"
                    csv_path = tables_dir / filename
                    xlsx_path = csv_path.with_suffix(".xlsx")
                    _write_generated_csv(frame, csv_path)
                    _write_generated_xlsx(frame, xlsx_path, sheet_name=f"hier_{stage_value}_{cluster_value}_{level}_{direction}")
                    key = f"{stage_value}::{cluster_value}::{level}::{direction}"
                    table_payloads[key] = {
                        "stage": stage_value,
                        "cluster": cluster_value,
                        "level": level,
                        "direction": direction,
                        "table": _frame_table_payload(frame),
                        "download_path": _relpath(xlsx_path, bundle_dir),
                        "download_label": "Download Excel",
                        "csv_download_path": _relpath(csv_path, bundle_dir),
                    }
                    level_directions.append(direction)
                if level_directions:
                    available_levels.append(level)
                    directions_by_level[level] = level_directions
            if available_levels:
                clusters_out.append(
                    {
                        "value": cluster_value,
                        "levels": available_levels,
                        "directions_by_level": directions_by_level,
                    }
                )
        if clusters_out:
            serialized_stages.append({"value": stage_value, "clusters": clusters_out})

    return {
        "available": bool(serialized_stages),
        "stages": serialized_stages,
        "tables": table_payloads,
    }


def _perturbation_mode_title(mode: str) -> str:
    titles = {
        "pathway": "Pathway Perturbation",
        "drug": "Drug Perturbation",
        "single_gene": "Single Gene Perturbation",
        "gene_combo": "Gene Combination Perturbation",
    }
    return titles.get(str(mode), _humanize_entry_name(mode))


def _load_grouped_perturbation_table(
    analysis_dir: Path,
    adata,
    *,
    mode: str,
    perturb_change: str,
    track: str,
    generated_tables: dict[str, str],
    precomputed_group_values: set[str],
) -> tuple[pd.DataFrame, str | None]:
    if str(track) == "overall" or str(track) in precomputed_group_values:
        frame = extract_perturbation_table(
            adata,
            mode=mode,
            perturb_change=float(str(perturb_change)),
            track=str(track),
        )
        return frame, None
    table_path = generated_tables.get(str(track), "")
    if not table_path:
        table_path = _ensure_overall_track_group_perturbation_table(
            analysis_dir,
            mode=mode,
            perturb_change=str(perturb_change),
            track_group=str(track),
        )
    return pd.read_csv(table_path), table_path


def _write_perturbation_bundle_assets(
    frame: pd.DataFrame,
    *,
    mode: str,
    asset_name: str,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, Any]:
    asset_stem = _bundle_download_stem(download_prefix, asset_name)
    table_dir = bundle_dir / "assets" / "perturbation_tables"
    figure_dir = bundle_dir / "assets" / "perturbation_figures"
    publication_dir = bundle_dir / "assets" / "publication"

    csv_path = table_dir / f"{asset_stem}.csv"
    xlsx_path = table_dir / f"{asset_stem}.xlsx"
    png_path = figure_dir / f"{asset_stem}.png"
    pdf_path = publication_dir / f"{asset_stem}.pdf"

    _write_generated_csv(frame, csv_path)
    _write_generated_xlsx(frame, xlsx_path, sheet_name=asset_stem)

    plot_table = frame.head(OFFLINE_PERTURBATION_TOP_N).copy()
    use_pval = str(mode) != "pathway"
    fig, _ax, _work = plot_perturbation_ranked_bar(
        plot_table,
        top_n=min(OFFLINE_PERTURBATION_TOP_N, len(plot_table)),
        use_neglog10_pval=use_pval,
        show=False,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "table_download_path": _relpath(xlsx_path, bundle_dir),
        "table_download_label": "Download detailed table (Excel)",
        "table_csv_path": _relpath(csv_path, bundle_dir),
        "plot_path": _relpath(png_path, bundle_dir),
        "plot_download_path": _relpath(pdf_path, bundle_dir),
    }


def _load_stage_specific_grouped_perturbation_table(
    analysis_dir: Path,
    *,
    mode: str,
    stage: str,
    track: str,
    stage_mode_entry: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    if str(track) == "overall":
        table_path = str((stage_mode_entry.get("tables_by_stage") or {}).get(str(stage)) or "")
        if not table_path:
            raise KeyError(f"Missing stage-specific perturbation table for mode={mode!r}, stage={stage!r}.")
        return pd.read_csv(table_path), table_path
    table_path = str((((stage_mode_entry.get("track_tables_by_stage") or {}).get(str(stage)) or {}).get(str(track))) or "")
    if not table_path:
        table_path = _ensure_stage_specific_track_group_perturbation_table(
            analysis_dir,
            mode=mode,
            stage=str(stage),
            track_group=str(track),
        )
    return pd.read_csv(table_path), table_path


def _build_perturbation_bundle(
    analysis_dir: Path,
    adata,
    *,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, Any]:
    metadata = _extract_metadata(adata, analysis_dir=analysis_dir)
    perturbation_section = metadata.get("perturbation") or {}
    standard_modes = perturbation_section.get("modes") or {}
    stage_specific_modes = perturbation_section.get("stage_specific_modes") or {}
    stages = list((metadata.get("umap") or {}).get("stages") or [])

    tables: dict[str, Any] = {}
    modes_out: list[dict[str, Any]] = []

    for mode in ("pathway", "drug"):
        uns_key = PERTURBATION_MODE_KEYS.get(mode)
        payload = adata.uns.get(uns_key) if uns_key else None
        stage_mode_entry = stage_specific_modes.get(mode) or {}
        has_overall = isinstance(payload, dict) and bool(payload)
        has_stage_specific = bool(stage_mode_entry.get("tables_by_stage"))
        if not has_overall and not has_stage_specific:
            continue

        scope_options: list[str] = []
        if has_overall:
            scope_options.append("overall")
        if has_stage_specific:
            scope_options.append("stage")
        perturb_changes_out: list[str] = []
        tracks_by_change: dict[str, list[dict[str, str]]] = {}
        stage_options: list[dict[str, str]] = []
        resolved_stage_options: list[dict[str, str]] = []
        tracks_by_stage: dict[str, list[dict[str, str]]] = {}

        if has_overall:
            for perturb_change in _sorted_keys(payload.keys()):
                track_bucket = payload.get(perturb_change)
                if not isinstance(track_bucket, dict):
                    continue
                available_track_options = (standard_modes.get(mode, {}).get("tracks_by_change", {}) or {}).get(
                    str(perturb_change),
                    [],
                )
                raw_non_overall_values = [
                    str(item.get("value"))
                    for item in available_track_options
                    if str(item.get("value")) != "overall"
                ]
                normalized_non_overall_values = [
                    _normalize_stage_track_group_value(value)
                    for value in raw_non_overall_values
                ]
                precomputed_group_values: set[str] = set()
                if raw_non_overall_values and (
                    len(set(raw_non_overall_values)) == len(set(normalized_non_overall_values))
                    and set(raw_non_overall_values) == set(normalized_non_overall_values)
                ):
                    precomputed_group_values = set(raw_non_overall_values)

                generated_track_tables = _discover_overall_track_group_perturbation_tables(
                    analysis_dir,
                    mode=mode,
                    perturb_change=str(perturb_change),
                )
                if not precomputed_group_values and not generated_track_tables:
                    try:
                        generated_track_tables = _generate_overall_track_group_perturbation_tables(
                            analysis_dir,
                            mode=mode,
                            perturb_change=str(perturb_change),
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
                    stages=stages,
                    track_values=track_values,
                )
                if not track_options:
                    continue

                resolved_track_options: list[dict[str, str]] = []
                for option in track_options:
                    track_value = str(option.get("value") or "")
                    try:
                        frame, _table_path = _load_grouped_perturbation_table(
                            analysis_dir,
                            adata,
                            mode=mode,
                            perturb_change=str(perturb_change),
                            track=track_value,
                            generated_tables=generated_track_tables,
                            precomputed_group_values=precomputed_group_values,
                        )
                    except Exception:
                        continue
                    if frame.empty:
                        continue
                    assets = _write_perturbation_bundle_assets(
                        frame,
                        mode=mode,
                        asset_name=f"perturbation_{mode}_scope_overall_change_{perturb_change}_{option.get('label') or track_value}",
                        bundle_dir=bundle_dir,
                        download_prefix=download_prefix,
                    )
                    key = f"{mode}::overall::{perturb_change}::{track_value}"
                    tables[key] = {
                        "scope": "overall",
                        "mode": mode,
                        "perturb_change": str(perturb_change),
                        "track": track_value,
                        "track_label": str(option.get("label") or track_value),
                        **assets,
                    }
                    resolved_track_options.append({"value": track_value, "label": str(option.get("label") or track_value)})

                if resolved_track_options:
                    perturb_changes_out.append(str(perturb_change))
                    tracks_by_change[str(perturb_change)] = resolved_track_options

        if has_stage_specific:
            stage_options = list(stage_mode_entry.get("stage_options") or [])
            for stage_option in stage_options:
                stage_value = str(stage_option.get("value") or "")
                if not stage_value:
                    continue
                representative_tracks: dict[str, str] = {}
                for item in metadata.get("dynamic_markers", {}).get("tracks", []):
                    raw_value = str(item.get("value") or "")
                    if raw_value:
                        representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)
                standard_mode_entry = standard_modes.get(mode) or {}
                for track_list in (standard_mode_entry.get("tracks_by_change") or {}).values():
                    for item in track_list:
                        raw_value = str(item.get("value") or "")
                        if not raw_value or raw_value == "overall":
                            continue
                        representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)
                for raw_value in _sorted_keys(((stage_mode_entry.get("track_tables_by_stage") or {}).get(stage_value) or {}).keys()):
                    if raw_value:
                        representative_tracks.setdefault(_normalize_stage_track_group_value(raw_value), raw_value)

                stage_track_values = ["overall"]
                stage_track_values.extend(
                    _sorted_keys(key for key in representative_tracks.keys() if key != "overall")
                )
                track_options = _build_stage_track_options(
                    adata,
                    stages=stages,
                    track_values=stage_track_values,
                )
                if not track_options:
                    continue
                resolved_track_options: list[dict[str, str]] = []
                for option in track_options:
                    track_value = str(option.get("value") or "")
                    try:
                        frame, _table_path = _load_stage_specific_grouped_perturbation_table(
                            analysis_dir,
                            mode=mode,
                            stage=stage_value,
                            track=track_value,
                            stage_mode_entry=stage_mode_entry,
                        )
                    except Exception:
                        continue
                    if frame.empty:
                        continue
                    assets = _write_perturbation_bundle_assets(
                        frame,
                        mode=mode,
                        asset_name=f"perturbation_{mode}_scope_stage_{stage_value}_{option.get('label') or track_value}",
                        bundle_dir=bundle_dir,
                        download_prefix=download_prefix,
                    )
                    key = f"{mode}::stage::{stage_value}::{track_value}"
                    tables[key] = {
                        "scope": "stage",
                        "mode": mode,
                        "stage": stage_value,
                        "stage_label": str(stage_option.get("label") or stage_value),
                        "track": track_value,
                        "track_label": str(option.get("label") or track_value),
                        **assets,
                    }
                    resolved_track_options.append({"value": track_value, "label": str(option.get("label") or track_value)})

                if resolved_track_options:
                    tracks_by_stage[stage_value] = resolved_track_options
                    resolved_stage_options.append(
                        {"value": stage_value, "label": str(stage_option.get("label") or stage_value)}
                    )

        if perturb_changes_out or resolved_stage_options:
            modes_out.append(
                {
                    "value": mode,
                    "title": _perturbation_mode_title(mode),
                    "scope_options": scope_options,
                    "perturb_changes": perturb_changes_out,
                    "tracks_by_change": tracks_by_change,
                    "stage_options": resolved_stage_options,
                    "tracks_by_stage": tracks_by_stage,
                }
            )

    return {
        "available": bool(modes_out),
        "modes": modes_out,
        "tables": tables,
    }


def _collect_entries(
    visual_manifest: Mapping[str, Any],
    *,
    bundle_dir: Path,
    analysis_dir: Path,
) -> list[dict[str, Any]]:
    used_names: set[str] = set()
    figures_dir = bundle_dir / "assets" / "figures"
    tables_dir = bundle_dir / "assets" / "tables"
    misc_dir = bundle_dir / "assets" / "misc"
    entries: list[dict[str, Any]] = []
    for success in visual_manifest.get("successes", []) or []:
        name = str(success.get("name") or "artifact")
        entry = {
            "name": name,
            "title": _humanize_entry_name(name),
            "artifacts": [],
        }
        for key in ("file", "table", "matrix", "nodes", "edges"):
            raw = success.get(key)
            if not raw:
                continue
            src = Path(raw)
            if not src.exists():
                continue
            destination_root = figures_dir if key == "file" and _artifact_type(src) == "image" else tables_dir if _artifact_type(src) == "csv" else misc_dir
            copied = Path(_copy_artifact(src, destination_root, used_names))
            artifact = {
                "label": key,
                "type": _artifact_type(copied),
                "path": _relpath(copied, bundle_dir),
            }
            if artifact["type"] == "csv":
                artifact["preview"] = _csv_preview(src)
                try:
                    frame = pd.read_csv(src)
                    xlsx_path = _reserve_unique_artifact_path(copied.with_suffix(".xlsx"), used_names)
                    _write_generated_xlsx(frame, xlsx_path, sheet_name=name)
                    artifact["download_path"] = _relpath(xlsx_path, bundle_dir)
                    artifact["download_label"] = "Download Excel"
                    artifact["csv_download_path"] = _relpath(copied, bundle_dir)
                except Exception:
                    artifact["download_label"] = "Download CSV"
            entry["artifacts"].append(artifact)
        if name in SHARE_BUNDLE_PUBLICATION_EXPORTS:
            try:
                pdf_bytes, filename = export_static_publication_pdf(analysis_dir, name)
            except Exception:
                pdf_bytes = None
                filename = None
            if pdf_bytes and filename:
                destination = _reserve_unique_artifact_path(bundle_dir / "assets" / "publication" / filename, used_names)
                destination.write_bytes(pdf_bytes)
                for artifact in entry["artifacts"]:
                    if str(artifact.get("type") or "") == "image":
                        artifact["download_path"] = _relpath(destination, bundle_dir)
                        artifact["download_label"] = "Download PDF"
                        break
        for raw in success.get("files", []) or []:
            src = Path(raw)
            if not src.exists():
                continue
            copied = Path(_copy_artifact(src, misc_dir, used_names))
            entry["artifacts"].append(
                {
                    "label": "file",
                    "type": _artifact_type(copied),
                    "path": _relpath(copied, bundle_dir),
                }
            )
        if entry["artifacts"]:
            entries.append(entry)
    return entries


def _override_stage_umap_entry(
    entries: list[dict[str, Any]],
    *,
    bundle_dir: Path,
    adata,
) -> None:
    target_entry = next((entry for entry in entries if str(entry.get("name") or "") == "stage_umap_grid"), None)
    if target_entry is None:
        return
    artifacts = list(target_entry.get("artifacts") or [])
    existing_image_artifact = next(
        (
            artifact
            for artifact in artifacts
            if str(artifact.get("type") or "") == "image" and str(artifact.get("path") or "").strip()
        ),
        None,
    )
    if existing_image_artifact is not None:
        src = bundle_dir / str(existing_image_artifact.get("path"))
        if src.exists():
            output_path = bundle_dir / "assets" / "figures" / "stage_umap_grid_300dpi.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, output_path)
            existing_image_artifact["path"] = _relpath(output_path, bundle_dir)
            target_entry["artifacts"] = artifacts
            return

    try:
        fig, _axes = plot_stage_umap_grid(adata, show=False)
        output_path = bundle_dir / "assets" / "figures" / "stage_umap_grid_300dpi.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return

    replaced = False
    for artifact in artifacts:
        if str(artifact.get("type") or "") == "image":
            artifact["path"] = _relpath(output_path, bundle_dir)
            replaced = True
            break
    if not replaced:
        artifacts.insert(
            0,
            {
                "label": "file",
                "type": "image",
                "path": _relpath(output_path, bundle_dir),
            },
        )
    target_entry["artifacts"] = artifacts


def _build_temporal_graph_payload(analysis_dir: Path, adata) -> dict[str, Any] | None:
    if "edges" not in adata.uns:
        fallback = _load_temporal_edges_fallback(analysis_dir)
        if fallback is not None:
            adata.uns["edges"] = fallback
    if "edges" not in adata.uns:
        return None
    fig, _ax, nodes, edges = plot_temporal_dynamics_graph(adata, show=False)
    plt.close(fig)

    node_lookup = {
        row["node"]: row
        for _, row in nodes.iterrows()
    }
    stage_order = [str(value) for value in nodes["stage"].astype(str).tolist()]
    deduped_stage_order: list[str] = []
    for stage in stage_order:
        if stage not in deduped_stage_order:
            deduped_stage_order.append(stage)
    stage_tick_values: list[float] = []
    stage_tick_labels: list[str] = []
    for stage in deduped_stage_order:
        stage_nodes = nodes.loc[nodes["stage"].astype(str) == str(stage)]
        if stage_nodes.empty:
            continue
        stage_tick_values.append(float(stage_nodes["y"].astype(float).mean()))
        stage_tick_labels.append(f"Stage {stage}" if not str(stage).lower().startswith("stage") else str(stage))

    payload_nodes: list[dict[str, Any]] = []
    for index, row in nodes.reset_index(drop=True).iterrows():
        payload = {
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
        if "root_stage0_cluster" in row and str(row["root_stage0_cluster"]).strip():
            payload["root_stage0_cluster"] = str(row["root_stage0_cluster"])
        if "root_stage0_cell_type" in row and str(row["root_stage0_cell_type"]).strip():
            payload["root_stage0_cell_type"] = str(row["root_stage0_cell_type"])
        if "track_label" in row and str(row["track_label"]).strip():
            payload["track_label"] = str(row["track_label"])
        if "track_key" in row and str(row["track_key"]).strip():
            payload["track_key"] = str(row["track_key"])
        payload_nodes.append(payload)

    payload_edges: list[dict[str, Any]] = []
    for _, row in edges.iterrows():
        src = node_lookup.get(row["src_node"])
        dst = node_lookup.get(row["dst_node"])
        if src is None or dst is None:
            continue
        payload = {
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
            payload["track_label"] = str(row["track_label"])
        payload_edges.append(payload)

    return {
        "node_count": len(payload_nodes),
        "edge_count": len(payload_edges),
        "stage_order": deduped_stage_order,
        "stage_tick_values": stage_tick_values,
        "stage_tick_labels": stage_tick_labels,
        "nodes": payload_nodes,
        "edges": payload_edges,
    }


def _write_track_enrichment_exports(
    summary: Mapping[str, Any],
    *,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, str]:
    track_key = _safe_name(str(summary.get("track_key") or "track"))
    track_label = str(summary.get("track_label") or summary.get("track_key") or "track")
    out: dict[str, str] = {}
    for path in summary.get("paths", []) or []:
        path_key = str(path.get("path_key") or "").strip()
        table = path.get("enrichment") or {}
        rows = table.get("rows") or []
        columns = table.get("columns") or []
        if not path_key or not columns:
            continue
        try:
            frame = pd.DataFrame(rows, columns=columns)
        except Exception:
            continue
        path_label = str(path.get("label") or path_key)
        filename = _bundle_download_stem(
            download_prefix,
            "track_enrichment",
            track_label,
            f"path_{path_label}",
            "details",
        ) + ".csv"
        csv_output = bundle_dir / "assets" / "track_tables" / filename
        xlsx_output = csv_output.with_suffix(".xlsx")
        _write_generated_csv(frame, csv_output)
        _write_generated_xlsx(frame, xlsx_output, sheet_name=f"{track_key}_{path_key}")
        out[path_key] = _relpath(xlsx_output, bundle_dir)
    return out


def _sanitize_track_summary(summary: Mapping[str, Any], *, enrichment_downloads: Mapping[str, str] | None = None) -> dict[str, Any]:
    downloads = dict(enrichment_downloads or {})
    return {
        "track_key": str(summary.get("track_key") or ""),
        "track_label": str(summary.get("track_label") or summary.get("track_key") or ""),
        "path_labels": list(summary.get("path_labels") or []),
        "selected_path": summary.get("selected_path"),
        "path_plot": summary.get("path_plot") or {"stage_labels": [], "paths": []},
        "paths": [
            {
                **dict(path),
                "enrichment_download_path": downloads.get(str(path.get("path_key") or "")),
            }
            for path in list(summary.get("paths") or [])
        ],
        "original_html_path": str(summary.get("original_html_path") or ""),
    }


def _build_track_summaries(
    analysis_dir: Path,
    temporal_graph: Mapping[str, Any] | None,
    *,
    bundle_dir: Path,
    download_prefix: str,
) -> dict[str, Any]:
    track_keys: list[str] = []
    if temporal_graph is not None:
        for node in temporal_graph.get("nodes", []) or []:
            track_key = str(node.get("track_key") or "").strip()
            if track_key and track_key not in track_keys:
                track_keys.append(track_key)
    idrem_dir = analysis_dir / "idrem"
    if not track_keys and idrem_dir.exists():
        for track_dir in sorted(idrem_dir.glob("*.txt_viz")):
            key = track_dir.name[: -len(".txt_viz")]
            if key not in track_keys:
                track_keys.append(key)
    cache = IdremTrackSummaryCache()
    summaries: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for track_key in track_keys:
        try:
            summary = cache.get_summary(analysis_dir, track_key=track_key, track_label=track_key)
        except Exception as exc:
            errors[track_key] = f"{type(exc).__name__}: {exc}"
            continue
        enrichment_downloads = _write_track_enrichment_exports(
            summary,
            bundle_dir=bundle_dir,
            download_prefix=download_prefix,
        )
        summaries[track_key] = _sanitize_track_summary(summary, enrichment_downloads=enrichment_downloads)
    return {"tracks": summaries, "errors": errors}


def _entry_map(entries: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(entry.get("name") or ""): dict(entry) for entry in entries}


def _section_for_entry(name: str) -> str:
    if name in OVERVIEW_ENTRY_NAMES:
        return "overview"
    if name in TEMPORAL_ENTRY_NAMES:
        return "temporal"
    if name in PERTURBATION_ENTRY_NAMES:
        return "perturbation"
    if name in DYNAMIC_ENTRY_NAMES:
        return "dynamic"
    if name in HIERARCHICAL_ENTRY_NAMES:
        return "hierarchical"
    return "other"


def _render_artifact_preview(artifact: Mapping[str, Any]) -> str:
    artifact_type = str(artifact.get("type") or "file")
    path = html.escape(str(artifact.get("path") or ""))
    if artifact_type == "image":
        return f'<img class="artifact-image" src="{path}" alt="{path}">'
    if artifact_type == "csv":
        preview = artifact.get("preview") or {}
        columns = preview.get("columns") or []
        rows = preview.get("rows") or []
        if not columns:
            return ""
        thead = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
        body_rows = []
        for row in rows:
            body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
        return (
            '<div class="table-wrap"><table class="report-table">'
            f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        )
    return ""


def _render_entry_cards(entries: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for entry in entries:
        title = html.escape(str(entry.get("title") or entry.get("name") or "Artifact"))
        artifact_parts: list[str] = []
        for artifact in entry.get("artifacts", []) or []:
            label = html.escape(str(artifact.get("label") or "file"))
            path = html.escape(str(artifact.get("path") or ""))
            download_path = html.escape(str(artifact.get("download_path") or artifact.get("path") or ""))
            download_label = html.escape(str(artifact.get("download_label") or "Download"))
            artifact_parts.append(
                "<article class=\"artifact-card\">"
                f"<div class=\"artifact-head\"><strong>{label}</strong><a href=\"{download_path}\" download>{download_label}</a></div>"
                f"{_render_artifact_preview(artifact)}"
                "</article>"
            )
        if artifact_parts:
            parts.append(
                "<section class=\"report-card\">"
                f"<h3>{title}</h3>"
                f"{''.join(artifact_parts)}"
                "</section>"
            )
    return "".join(parts)


def _perturbation_mode_groups(entries: Iterable[Mapping[str, Any]], *, skip_modes: Iterable[str] = ()) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    skip = {str(mode) for mode in skip_modes}
    title_map = {
        "pathway": "Pathway Perturbation",
        "drug": "Drug Perturbation",
        "single_gene": "Single Gene Perturbation",
        "gene_combo": "Gene Combination Perturbation",
    }
    for entry in entries:
        name = str(entry.get("name") or "")
        mode = None
        if name.endswith("_ranked_bar"):
            mode = name[: -len("_ranked_bar")]
        elif name.endswith("_perturbation_table"):
            mode = name[: -len("_perturbation_table")]
        if not mode:
            continue
        if mode in skip:
            continue
        group = groups.setdefault(
            mode,
            {
                "mode": mode,
                "title": title_map.get(mode, _humanize_entry_name(mode)),
                "plot_artifact": None,
                "table_artifact": None,
            },
        )
        for artifact in entry.get("artifacts", []) or []:
            artifact_type = str(artifact.get("type") or "")
            if name.endswith("_ranked_bar") and artifact_type == "image" and group["plot_artifact"] is None:
                group["plot_artifact"] = dict(artifact)
            if name.endswith("_perturbation_table") and artifact_type == "csv" and group["table_artifact"] is None:
                group["table_artifact"] = dict(artifact)
    ordered_modes = ["pathway", "drug", "single_gene", "gene_combo"]
    return [groups[mode] for mode in ordered_modes if mode in groups]


def _render_offline_perturbation_cards(perturbation_bundle: Mapping[str, Any]) -> str:
    if not perturbation_bundle.get("available"):
        return ""
    cards: list[str] = []
    for mode_entry in perturbation_bundle.get("modes", []) or []:
        mode_value = str(mode_entry.get("value") or "")
        safe_mode = html.escape(mode_value)
        title = html.escape(str(mode_entry.get("title") or _perturbation_mode_title(mode_value)))
        cards.append(
            "<section class=\"report-card marker-panel\">"
            f"<h3>{title}</h3>"
            "<div class=\"control-grid\">"
            f"<div class=\"control\" id=\"perturb-scope-control-{safe_mode}\"><label for=\"perturb-scope-{safe_mode}\">Scope</label><select id=\"perturb-scope-{safe_mode}\"></select></div>"
            f"<div class=\"control\"><label for=\"perturb-change-{safe_mode}\">Perturb change</label><select id=\"perturb-change-{safe_mode}\"></select></div>"
            f"<div class=\"control\" id=\"perturb-stage-control-{safe_mode}\"><label for=\"perturb-stage-{safe_mode}\">Stage</label><select id=\"perturb-stage-{safe_mode}\"></select></div>"
            f"<div class=\"control\"><label for=\"perturb-track-{safe_mode}\">Track</label><select id=\"perturb-track-{safe_mode}\"></select></div>"
            "</div>"
            f"<img id=\"perturb-plot-{safe_mode}\" class=\"artifact-image\" alt=\"{title}\">"
            f"<div id=\"perturb-summary-{safe_mode}\" class=\"graph-help\"></div>"
            "<div class=\"download-links\">"
            f"<a id=\"perturb-plot-download-{safe_mode}\" href=\"#\" download>Download plot (PDF)</a>"
            f"<a id=\"perturb-table-download-{safe_mode}\" href=\"#\" download>Download detailed table (Excel)</a>"
            "</div>"
            "</section>"
        )
    return "".join(cards)


def _render_perturbation_section(
    entries: Iterable[Mapping[str, Any]],
    perturbation_bundle: Mapping[str, Any],
) -> str:
    cards: list[str] = []
    bundle_modes = {str(item.get("value") or "") for item in perturbation_bundle.get("modes", []) or []}
    offline_cards = _render_offline_perturbation_cards(perturbation_bundle)
    if offline_cards:
        cards.append(offline_cards)
    for group in _perturbation_mode_groups(entries, skip_modes=bundle_modes):
        plot_artifact = group.get("plot_artifact")
        table_artifact = group.get("table_artifact")
        actions: list[str] = []
        body = '<p class="empty">No perturbation plot was exported for this mode.</p>'
        if plot_artifact is not None:
            plot_path = html.escape(str(plot_artifact.get("path") or ""))
            plot_download_path = html.escape(str(plot_artifact.get("download_path") or plot_artifact.get("path") or ""))
            plot_download_label = html.escape(str(plot_artifact.get("download_label") or "Download plot"))
            body = f'<img class="artifact-image" src="{plot_path}" alt="{plot_path}">'
            actions.append(f'<a href="{plot_download_path}" download>{plot_download_label}</a>')
        if table_artifact is not None:
            table_download_path = html.escape(str(table_artifact.get("download_path") or table_artifact.get("path") or ""))
            table_download_label = html.escape(str(table_artifact.get("download_label") or "Download detailed table (Excel)"))
            actions.append(f'<a href="{table_download_path}" download>{table_download_label}</a>')
        cards.append(
            "<section class=\"report-card\">"
            f"<h3>{html.escape(str(group.get('title') or 'Perturbation'))}</h3>"
            f"{body}"
            f"<div class=\"download-links\">{''.join(actions)}</div>"
            "</section>"
        )
    if not cards:
        return '<p class="empty">No perturbation outputs are included in this bundle.</p>'
    return "".join(cards)


def _render_download_catalog(entries: Iterable[Mapping[str, Any]]) -> str:
    items: list[str] = []
    for entry in entries:
        title = html.escape(str(entry.get("title") or entry.get("name") or "Artifact"))
        links: list[str] = []
        for artifact in entry.get("artifacts", []) or []:
            path = html.escape(str(artifact.get("download_path") or artifact.get("path") or ""))
            label = html.escape(
                str(
                    artifact.get("download_label")
                    or artifact.get("label")
                    or artifact.get("type")
                    or "file"
                )
            )
            links.append(f'<a href="{path}" download>{label}</a>')
        if links:
            items.append(
                "<div class=\"download-row\">"
                f"<strong>{title}</strong>"
                f"<div class=\"download-links\">{''.join(links)}</div>"
                "</div>"
            )
    if not items:
        return '<p class="empty">No exported artifacts were copied into this bundle.</p>'
    return "".join(items)


def _render_report_html(
    *,
    bundle_title: str,
    generated_at: str,
    entries: list[dict[str, Any]],
    perturbation_bundle: Mapping[str, Any],
    temporal_graph: Mapping[str, Any] | None,
    track_summaries: Mapping[str, Any],
    dynamic_markers: Mapping[str, Any],
    hierarchical_markers: Mapping[str, Any],
) -> str:
    overview_entries = [entry for entry in entries if _section_for_entry(str(entry.get("name") or "")) == "overview"]
    perturbation_entries = [entry for entry in entries if _section_for_entry(str(entry.get("name") or "")) == "perturbation"]
    entry_map = _entry_map(entries)
    temporal_downloads = []
    temporal_entry = entry_map.get("temporal_dynamics_graph")
    if temporal_entry:
        for artifact in temporal_entry.get("artifacts", []) or []:
            download_path = str(artifact.get("download_path") or artifact.get("path") or "")
            download_label = str(
                artifact.get("download_label")
                or artifact.get("label")
                or "file"
            )
            temporal_downloads.append(
                f'<a href="{html.escape(download_path)}" download>{html.escape(download_label)}</a>'
            )

    graph_json = json.dumps(temporal_graph or {}, separators=(",", ":")).replace("</script>", "<\\/script>")
    summary_json = json.dumps(track_summaries, separators=(",", ":")).replace("</script>", "<\\/script>")
    perturbation_json = json.dumps(perturbation_bundle or {}, separators=(",", ":")).replace("</script>", "<\\/script>")
    dynamic_json = json.dumps(dynamic_markers or {}, separators=(",", ":")).replace("</script>", "<\\/script>")
    hierarchical_json = json.dumps(hierarchical_markers or {}, separators=(",", ":")).replace("</script>", "<\\/script>")

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(bundle_title)}</title>
    <style>
      :root {{
        --bg: #f7f3ea;
        --panel: #fffdf8;
        --ink: #203036;
        --muted: #617079;
        --line: rgba(32, 48, 54, 0.14);
        --accent: #9d5c3f;
        --accent-soft: rgba(157, 92, 63, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(157, 92, 63, 0.08), transparent 32%),
          linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
      }}
      .page {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 80px; }}
      .hero {{ display: flex; gap: 24px; justify-content: space-between; align-items: flex-start; margin-bottom: 28px; }}
      .hero h1 {{ margin: 0 0 10px; font-size: 2.3rem; line-height: 1.08; }}
      .muted {{ color: var(--muted); }}
      .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 28px; }}
      .nav a, .pill {{
        display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px;
        text-decoration: none; color: var(--ink); background: rgba(255,255,255,0.72); border: 1px solid var(--line);
      }}
      .notice {{ padding: 14px 16px; border-radius: 18px; background: var(--accent-soft); border: 1px solid rgba(157,92,63,0.18); margin-bottom: 28px; }}
      .section {{ margin-top: 38px; }}
      .section h2 {{ margin: 0 0 14px; font-size: 1.45rem; }}
      .section-grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
      .stack-grid {{ display: grid; gap: 18px; grid-template-columns: 1fr; }}
      .report-card, .graph-shell {{
        background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 18px;
        box-shadow: 0 18px 40px rgba(41, 36, 25, 0.05);
      }}
      .control-grid {{
        display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        margin-bottom: 16px;
      }}
      .control label {{
        display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px;
      }}
      .control select {{
        width: 100%; padding: 10px 12px; border-radius: 14px; border: 1px solid var(--line);
        background: #fff; color: var(--ink);
      }}
      .report-card h3 {{ margin: 0 0 14px; font-size: 1.05rem; }}
      .artifact-card {{ border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }}
      .artifact-card:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
      .artifact-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
      .artifact-head a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
      .artifact-image {{ display: block; width: 100%; height: auto; border-radius: 16px; border: 1px solid var(--line); background: #fff; }}
      .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 14px; background: #fff; }}
      table {{ border-collapse: collapse; width: 100%; min-width: 420px; }}
      th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--line); font-size: 0.92rem; }}
      th {{ background: rgba(32,48,54,0.04); }}
      .report-table.enrichment-table {{ min-width: 0; table-layout: fixed; }}
      .report-table.enrichment-table th,
      .report-table.enrichment-table td {{ font-size: 0.84rem; line-height: 1.25; vertical-align: top; }}
      .report-table.enrichment-table th:nth-child(1),
      .report-table.enrichment-table td:nth-child(1) {{ width: 56px; }}
      .report-table.enrichment-table th:nth-child(3),
      .report-table.enrichment-table td:nth-child(3) {{ width: 132px; white-space: nowrap; }}
      .report-table.enrichment-table th:nth-child(2),
      .report-table.enrichment-table td:nth-child(2) {{ overflow-wrap: anywhere; word-break: break-word; }}
      .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
      .legend-item {{ display: inline-flex; align-items: center; gap: 8px; font-size: 0.92rem; color: var(--muted); }}
      .legend-swatch {{ width: 12px; height: 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.08); }}
      .graph-shell svg {{ width: 100%; height: auto; display: block; background: #fff; border-radius: 16px; border: 1px solid var(--line); }}
      .graph-help {{ margin-top: 12px; color: var(--muted); font-size: 0.95rem; }}
      .download-links {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }}
      .download-links a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
      .download-row {{ padding: 14px 0; border-top: 1px solid var(--line); }}
      .download-row:first-of-type {{ padding-top: 0; border-top: 0; }}
      .track-modal {{
        position: fixed; inset: 0; background: rgba(16, 24, 28, 0.52); display: none; align-items: center;
        justify-content: center; padding: 20px; z-index: 1000;
      }}
      .track-modal.visible {{ display: flex; }}
      .track-dialog {{
        width: min(1100px, 100%); max-height: 88vh; overflow: auto; background: var(--panel);
        border-radius: 24px; border: 1px solid var(--line); padding: 22px;
      }}
      .track-header {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 16px; }}
      .track-close {{
        border: 0; border-radius: 999px; background: rgba(32,48,54,0.08); color: var(--ink); cursor: pointer;
        padding: 10px 14px; font-weight: 600;
      }}
      .path-buttons {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
      .path-button {{
        border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--ink); cursor: pointer;
        padding: 8px 12px;
      }}
      .path-button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
      .path-plot {{ border: 1px solid var(--line); border-radius: 16px; background: #fff; padding: 10px; }}
      .path-grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-top: 18px; }}
      .path-table-card {{ border: 1px solid var(--line); border-radius: 18px; padding: 14px; background: #fff; }}
      .path-table-card h4 {{ margin: 0 0 10px; }}
      .path-table-card-wide {{ margin-top: 18px; }}
      .path-details-layout {{ display: grid; gap: 18px; }}
      .path-details-top {{ display: grid; gap: 18px; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; }}
      .track-section {{ border: 1px solid var(--line); border-radius: 18px; background: #fff; overflow: hidden; }}
      .track-section + .track-section {{ margin-top: 14px; }}
      .track-section summary {{
        cursor: pointer; list-style: none; padding: 14px 16px; font-weight: 600; user-select: none;
        background: rgba(32,48,54,0.03);
      }}
      .track-section summary::-webkit-details-marker {{ display: none; }}
      .track-section-body {{ padding: 14px; }}
      .enrichment-tabs {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
      .enrichment-tab {{
        border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--ink); cursor: pointer;
        padding: 8px 12px; font: inherit;
      }}
      .enrichment-tab.is-active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
      .enrichment-panel {{ display: none; }}
      .enrichment-panel.is-active {{ display: block; }}
      .chart-shell {{
        border: 1px solid var(--line); border-radius: 18px; background: #fff; padding: 10px;
        min-height: 220px;
      }}
      .marker-panel {{ display: grid; gap: 16px; }}
      .chart-note {{ margin: 0; color: var(--muted); font-size: 0.93rem; }}
      .empty {{ color: var(--muted); font-style: italic; }}
      @media (max-width: 760px) {{
        .hero {{ flex-direction: column; }}
        .page {{ padding: 22px 16px 60px; }}
        .path-details-top {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <div>
          <p class="muted">Offline Share Bundle</p>
          <h1>{html.escape(bundle_title)}</h1>
          <p class="muted">Generated {html.escape(generated_at)}. This bundle is designed to be unzipped and opened locally as a standalone report.</p>
        </div>
        <div class="pill">Static UMAPs + Clickable Temporal Graph</div>
      </section>

      <div class="notice">
        UMAPs and most figures are static in this offline export. The temporal dynamics graph remains clickable and opens precomputed track summaries directly inside this report.
      </div>

      <nav class="nav">
        <a href="#overview">Overview</a>
        <a href="#temporal-graph">Temporal Graph</a>
        <a href="#perturbation">Perturbation</a>
        <a href="#markers">Markers</a>
        <a href="#downloads">Downloads</a>
      </nav>

      <section id="overview" class="section">
        <h2>Overview</h2>
        <div class="section-grid">
          {_render_entry_cards(overview_entries)}
        </div>
      </section>

      <section id="temporal-graph" class="section">
        <h2>Temporal Dynamics Graph</h2>
        <div class="graph-shell">
          <div id="temporal-graph-container"></div>
          <div class="graph-help">Click a node with an associated track to open its offline iDREM summary.</div>
          <div class="download-links">{''.join(temporal_downloads)}</div>
          <div id="temporal-legend" class="legend"></div>
        </div>
      </section>

      <section id="perturbation" class="section">
        <h2>Perturbation</h2>
        <div class="section-grid">
          {_render_perturbation_section(perturbation_entries, perturbation_bundle)}
        </div>
      </section>

      <section id="markers" class="section">
        <h2>Marker Discovery</h2>
        <div class="stack-grid">
          <div class="report-card marker-panel">
            <h3>Dynamic Markers</h3>
            <div class="control-grid">
              <div class="control">
                <label for="dynamic-track">Temporal Track</label>
                <select id="dynamic-track"></select>
              </div>
              <div class="control">
                <label for="dynamic-direction">Direction</label>
                <select id="dynamic-direction"></select>
              </div>
            </div>
            <div id="dynamic-chart" class="chart-shell"></div>
            <div id="dynamic-table"></div>
            <div class="download-links">
              <a id="dynamic-download" href="#" download>Download dynamic marker table (Excel)</a>
            </div>
          </div>
          <div class="report-card marker-panel">
            <h3>Hierarchical Markers</h3>
            <div class="control-grid">
              <div class="control">
                <label for="hier-stage">Stage</label>
                <select id="hier-stage"></select>
              </div>
              <div class="control">
                <label for="hier-cluster">Cluster</label>
                <select id="hier-cluster"></select>
              </div>
              <div class="control">
                <label for="hier-level">Level</label>
                <select id="hier-level"></select>
              </div>
              <div class="control">
                <label for="hier-direction">Direction</label>
                <select id="hier-direction"></select>
              </div>
            </div>
            <div id="hier-chart" class="chart-shell"></div>
            <p id="hier-note" class="chart-note">Bars show signed log fold changes for the selected branch against its sibling comparison clusters.</p>
            <div id="hier-table"></div>
            <div class="download-links">
              <a id="hier-download" href="#" download>Download hierarchical marker table (Excel)</a>
            </div>
          </div>
        </div>
      </section>

      <section id="downloads" class="section">
        <h2>Bundle Contents</h2>
        <div class="report-card">
          <p class="muted">All copied figures and tables are included as relative files inside this bundle. Use the links below if you want the raw artifacts directly.</p>
          {_render_download_catalog(entries)}
        </div>
      </section>
    </div>

    <div id="track-modal" class="track-modal" aria-hidden="true">
      <div class="track-dialog">
        <div class="track-header">
          <div>
            <p class="muted">Temporal Track Summary</p>
            <h2 id="track-title">Track</h2>
          </div>
          <button id="track-close" class="track-close" type="button">Close</button>
        </div>
        <div id="track-plot" class="path-plot"></div>
        <div id="path-buttons" class="path-buttons"></div>
        <div id="path-details" class="path-grid"></div>
      </div>
    </div>

    <script id="unagi-share-temporal" type="application/json">{graph_json}</script>
    <script id="unagi-share-tracks" type="application/json">{summary_json}</script>
    <script id="unagi-share-perturbation" type="application/json">{perturbation_json}</script>
    <script id="unagi-share-dynamic" type="application/json">{dynamic_json}</script>
    <script id="unagi-share-hierarchical" type="application/json">{hierarchical_json}</script>
    <script>
      (function () {{
        const graphPayload = JSON.parse(document.getElementById("unagi-share-temporal").textContent || "{{}}");
        const trackPayload = JSON.parse(document.getElementById("unagi-share-tracks").textContent || "{{}}");
        const perturbationPayload = JSON.parse(document.getElementById("unagi-share-perturbation").textContent || "{{}}");
        const dynamicPayload = JSON.parse(document.getElementById("unagi-share-dynamic").textContent || "{{}}");
        const hierarchicalPayload = JSON.parse(document.getElementById("unagi-share-hierarchical").textContent || "{{}}");
        const container = document.getElementById("temporal-graph-container");
        const legend = document.getElementById("temporal-legend");
        const modal = document.getElementById("track-modal");
        const title = document.getElementById("track-title");
        const closeButton = document.getElementById("track-close");
        const pathButtons = document.getElementById("path-buttons");
        const pathDetails = document.getElementById("path-details");
        const trackPlot = document.getElementById("track-plot");
        const dynamicTrack = document.getElementById("dynamic-track");
        const dynamicDirection = document.getElementById("dynamic-direction");
        const dynamicChart = document.getElementById("dynamic-chart");
        const dynamicTable = document.getElementById("dynamic-table");
        const dynamicDownload = document.getElementById("dynamic-download");
        const hierStage = document.getElementById("hier-stage");
        const hierCluster = document.getElementById("hier-cluster");
        const hierLevel = document.getElementById("hier-level");
        const hierDirection = document.getElementById("hier-direction");
        const hierChart = document.getElementById("hier-chart");
        const hierNote = document.getElementById("hier-note");
        const hierTable = document.getElementById("hier-table");
        const hierDownload = document.getElementById("hier-download");

        function uniqueLegendItems(nodes) {{
          const seen = new Map();
          (nodes || []).forEach(function (node) {{
            const key = String(node.cell_type || "NA");
            if (!seen.has(key)) {{
              seen.set(key, String(node.color || "#6E7C80"));
            }}
          }});
          return Array.from(seen.entries());
        }}

        function renderLegend(nodes) {{
          uniqueLegendItems(nodes).forEach(function (entry) {{
            const item = document.createElement("div");
            item.className = "legend-item";
            const swatch = document.createElement("span");
            swatch.className = "legend-swatch";
            swatch.style.background = entry[1];
            const label = document.createElement("span");
            label.textContent = entry[0];
            item.appendChild(swatch);
            item.appendChild(label);
            legend.appendChild(item);
          }});
        }}

        function createSvg(tag) {{
          return document.createElementNS("http://www.w3.org/2000/svg", tag);
        }}

        function escapeHtml(value) {{
          return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }}

        function formatEnrichmentTerm(value) {{
          const raw = String(value || "");
          const prefixes = ["REACTOME_", "KEGG_", "WP_", "BIOCARTA_", "GOBP_", "GOMF_", "GOCC_"];
          let text = raw;
          for (let index = 0; index < prefixes.length; index += 1) {{
            const prefix = prefixes[index];
            if (text.startsWith(prefix)) {{
              text = prefix.slice(0, -1) + ": " + text.slice(prefix.length);
              break;
            }}
          }}
          return text.replace(/_/g, " ");
        }}

        function scaleFactory(minValue, maxValue, start, end) {{
          if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || maxValue === minValue) {{
            return function () {{ return (start + end) / 2; }};
          }}
          const span = end - start;
          return function (value) {{
            return start + ((value - minValue) / (maxValue - minValue)) * span;
          }};
        }}

        function renderTemporalGraph(payload) {{
          const nodes = payload.nodes || [];
          const edges = payload.edges || [];
          if (!nodes.length) {{
            container.innerHTML = '<p class="empty">Temporal graph data is not available in this export.</p>';
            return;
          }}
          const width = 980;
          const height = Math.max(420, (payload.stage_order || []).length * 140);
          const margin = {{ top: 40, right: 40, bottom: 50, left: 120 }};
          const xValues = nodes.map(function (node) {{ return Number(node.x); }});
          const yValues = nodes.map(function (node) {{ return Number(node.y); }});
          const xScale = scaleFactory(Math.min.apply(null, xValues), Math.max.apply(null, xValues), margin.left, width - margin.right);
          const yScale = scaleFactory(Math.min.apply(null, yValues), Math.max.apply(null, yValues), height - margin.bottom, margin.top);

          const svg = createSvg("svg");
          svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
          svg.setAttribute("role", "img");
          svg.setAttribute("aria-label", "Temporal dynamics graph");

          const background = createSvg("rect");
          background.setAttribute("x", "0");
          background.setAttribute("y", "0");
          background.setAttribute("width", String(width));
          background.setAttribute("height", String(height));
          background.setAttribute("fill", "#ffffff");
          svg.appendChild(background);

          (payload.stage_tick_values || []).forEach(function (tick, index) {{
            const y = yScale(Number(tick));
            const line = createSvg("line");
            line.setAttribute("x1", String(margin.left));
            line.setAttribute("x2", String(width - margin.right));
            line.setAttribute("y1", String(y));
            line.setAttribute("y2", String(y));
            line.setAttribute("stroke", "rgba(31,42,46,0.14)");
            line.setAttribute("stroke-width", "1");
            svg.appendChild(line);

            const label = createSvg("text");
            label.setAttribute("x", String(margin.left - 14));
            label.setAttribute("y", String(y + 4));
            label.setAttribute("text-anchor", "end");
            label.setAttribute("font-size", "12");
            label.setAttribute("fill", "#617079");
            label.textContent = (payload.stage_tick_labels || [])[index] || "";
            svg.appendChild(label);
          }});

          edges.forEach(function (edge) {{
            const line = createSvg("line");
            line.setAttribute("x1", String(xScale(Number(edge.x0))));
            line.setAttribute("y1", String(yScale(Number(edge.y0))));
            line.setAttribute("x2", String(xScale(Number(edge.x1))));
            line.setAttribute("y2", String(yScale(Number(edge.y1))));
            line.setAttribute("stroke", "rgba(106,113,118,0.55)");
            line.setAttribute("stroke-width", "1.6");
            svg.appendChild(line);
          }});

          const rootStage = Array.isArray(payload.stage_order) && payload.stage_order.length
            ? String(payload.stage_order[0])
            : "0";

          nodes.forEach(function (node) {{
            const circle = createSvg("circle");
            circle.setAttribute("cx", String(xScale(Number(node.x))));
            circle.setAttribute("cy", String(yScale(Number(node.y))));
            circle.setAttribute("r", String(Math.max(5, Math.min(14, Number(node.node_size || 0) / 42))));
            circle.setAttribute("fill", String(node.color || "#6E7C80"));
            circle.setAttribute("stroke", "#ffffff");
            circle.setAttribute("stroke-width", "1.5");
            if (node.track_key && trackPayload.tracks && trackPayload.tracks[node.track_key]) {{
              circle.style.cursor = "pointer";
              circle.addEventListener("click", function () {{
                openTrack(node.track_key, node.track_label || node.track_key);
              }});
            }}
            const titleNode = createSvg("title");
            titleNode.textContent = [
              `Stage: ${{node.stage}}`,
              `Cluster: ${{node.cluster}}`,
              `ident: ${{node.cell_type || "NA"}}`,
              `Cells: ${{node.n_cells}}`,
              node.root_stage0_cluster
                ? `Stage 0 root: ${{node.root_stage0_cell_type || "NA"}} | cluster ${{node.root_stage0_cluster}}`
                : "",
              node.track_label ? `Track: ${{node.track_label}}` : "",
            ].filter(Boolean).join("\\n");
            circle.appendChild(titleNode);
            svg.appendChild(circle);

            if (String(node.stage) === rootStage && node.root_stage0_cluster) {{
              const text = createSvg("text");
              text.setAttribute("x", String(xScale(Number(node.x))));
              text.setAttribute("y", String(yScale(Number(node.y)) - 20));
              text.setAttribute("text-anchor", "middle");
              text.setAttribute("font-size", "11");
              text.setAttribute("font-weight", "600");
              text.setAttribute("fill", "#172124");
              const ident = String(node.root_stage0_cell_type || node.cell_type || "NA");
              text.textContent = `${{ident}}-${{String(node.root_stage0_cluster)}}`;
              svg.appendChild(text);
            }}
          }});

          container.innerHTML = "";
          container.appendChild(svg);
        }}

        function renderSimpleTable(table) {{
          if (!table || !Array.isArray(table.columns) || !table.columns.length) {{
            return '<p class="empty">No rows available.</p>';
          }}
          const columns = table.columns.map(function (column) {{ return String(column); }});
          const isEnrichmentCompact = columns.length === 3
            && columns[0] === 'Rank'
            && columns[1] === 'Term Name'
            && columns[2] === 'Adjusted p-value';
          const tableClass = isEnrichmentCompact ? 'report-table enrichment-table' : 'report-table';
          const thead = '<thead><tr>' + columns.map(function (column) {{
            return `<th>${{escapeHtml(column)}}<\/th>`;
          }}).join('') + '</tr></thead>';
          const body = '<tbody>' + (table.rows || []).map(function (row) {{
            return '<tr>' + row.map(function (cell, cellIndex) {{
              if (isEnrichmentCompact && cellIndex === 1) {{
                const raw = String(cell);
                return `<td title="${{escapeHtml(raw)}}">${{escapeHtml(formatEnrichmentTerm(raw))}}<\/td>`;
              }}
              return `<td>${{escapeHtml(cell)}}<\/td>`;
            }}).join('') + '</tr>';
          }}).join('') + '</tbody>';
          return '<div class="table-wrap"><table class="' + tableClass + '">' + thead + body + '</table></div>';
        }}

        function compactEnrichmentTable(table) {{
          if (!table || !Array.isArray(table.columns) || !table.columns.length) {{
            return {{ columns: [], rows: [] }};
          }}
          const columns = table.columns.map(function (column) {{ return String(column); }});
          const termIndex = columns.findIndex(function (column) {{
            return column.toLowerCase() === 'term name' || column.toLowerCase() === 'go term name';
          }});
          const adjustedIndex = columns.findIndex(function (column) {{
            const lowered = column.toLowerCase();
            return lowered === 'corrected p-value' || lowered === 'adjusted p-value' || lowered === 'adj.p';
          }});
          if (termIndex < 0 || adjustedIndex < 0) {{
            return table;
          }}
          return {{
            columns: ['Rank', 'Term Name', 'Adjusted p-value'],
            rows: (table.rows || []).map(function (row, index) {{
              return [index + 1, row[termIndex], row[adjustedIndex]];
            }}),
          }};
        }}

        function splitEnrichmentTables(table) {{
          if (!table || !Array.isArray(table.columns) || !table.columns.length) {{
            return [];
          }}
          const columns = table.columns.map(function (column) {{ return String(column); }});
          const categoryIndex = columns.findIndex(function (column) {{
            return column.toLowerCase() === 'category';
          }});
          if (categoryIndex < 0) {{
            return [{{ key: 'all', label: 'Enrichment', table: compactEnrichmentTable(table) }}];
          }}
          const groups = [
            {{
              key: 'biological_process',
              label: 'Biological process',
              match: function (value) {{
                return String(value || '').toLowerCase().includes('biological process');
              }},
            }},
            {{
              key: 'pathways',
              label: 'Pathways',
              match: function (value) {{
                return String(value || '').toLowerCase().includes('pathway');
              }},
            }},
          ];
          const split = groups.map(function (group) {{
            const rows = (table.rows || []).filter(function (row) {{
              return group.match(row[categoryIndex]);
            }});
            return {{
              key: group.key,
              label: group.label,
              table: compactEnrichmentTable({{ columns: table.columns, rows: rows }}),
            }};
          }}).filter(function (group) {{
            return Array.isArray(group.table.rows) && group.table.rows.length;
          }});
          if (split.length) {{
            return split;
          }}
          return [{{ key: 'all', label: 'Enrichment', table: compactEnrichmentTable(table) }}];
        }}

        function compactDynamicTable(table) {{
          if (!table || !Array.isArray(table.columns) || !table.columns.length) {{
            return {{ columns: [], rows: [] }};
          }}
          const columns = table.columns.map(function (column) {{ return String(column); }});
          const keepIndexes = [];
          columns.forEach(function (column, index) {{
            const lowered = column.toLowerCase();
            if (lowered !== 'track' && lowered !== 'direction') {{
              keepIndexes.push(index);
            }}
          }});
          return {{
            columns: keepIndexes.map(function (index) {{ return columns[index]; }}),
            rows: (table.rows || []).map(function (row) {{
              return keepIndexes.map(function (index) {{ return row[index]; }});
            }}),
          }};
        }}

        function setSelectOptions(select, options, selectedValue) {{
          const values = options || [];
          const current = values.some(function (option) {{ return String(option.value) === String(selectedValue); }})
            ? String(selectedValue)
            : (values[0] ? String(values[0].value) : '');
          select.innerHTML = '';
          values.forEach(function (option) {{
            const element = document.createElement('option');
            element.value = String(option.value);
            element.textContent = String(option.label || option.value);
            if (String(option.value) === current) {{
              element.selected = true;
            }}
            select.appendChild(element);
          }});
          return current;
        }}

        function findPerturbationMode(modeValue) {{
          return (perturbationPayload.modes || []).find(function (item) {{
            return String(item.value) === String(modeValue);
          }});
        }}

        function renderPerturbationPanel(modeValue) {{
          const mode = findPerturbationMode(modeValue);
          const scopeSelect = document.getElementById(`perturb-scope-${{modeValue}}`);
          const scopeControl = document.getElementById(`perturb-scope-control-${{modeValue}}`);
          const changeSelect = document.getElementById(`perturb-change-${{modeValue}}`);
          const stageControl = document.getElementById(`perturb-stage-control-${{modeValue}}`);
          const stageSelect = document.getElementById(`perturb-stage-${{modeValue}}`);
          const trackSelect = document.getElementById(`perturb-track-${{modeValue}}`);
          const plot = document.getElementById(`perturb-plot-${{modeValue}}`);
          const summary = document.getElementById(`perturb-summary-${{modeValue}}`);
          const plotDownload = document.getElementById(`perturb-plot-download-${{modeValue}}`);
          const tableDownload = document.getElementById(`perturb-table-download-${{modeValue}}`);
          if (!mode || !scopeSelect || !changeSelect || !stageSelect || !trackSelect || !plot || !summary || !plotDownload || !tableDownload) {{
            return;
          }}

          const scopeOptions = (mode.scope_options || []).map(function (value) {{
            return {{
              value: value,
              label: value === 'stage' ? 'Individual stage' : 'Overall',
            }};
          }});
          const selectedScope = setSelectOptions(scopeSelect, scopeOptions, scopeSelect.value);
          if (scopeControl) {{
            scopeControl.style.display = scopeOptions.length > 1 ? '' : 'none';
          }}

          let key = '';
          let summaryText = '';
          if (selectedScope === 'stage') {{
            const stageOptions = (mode.stage_options || []).map(function (option) {{
              return {{ value: option.value, label: option.label || option.value }};
            }});
            const selectedStage = setSelectOptions(stageSelect, stageOptions, stageSelect.value);
            const trackOptions = (((mode.tracks_by_stage || {{}})[selectedStage]) || []).map(function (option) {{
              return {{ value: option.value, label: option.label || option.value }};
            }});
            const selectedTrack = setSelectOptions(trackSelect, trackOptions, trackSelect.value);
            if (stageControl) {{
              stageControl.style.display = '';
            }}
            if (changeSelect && changeSelect.parentElement) {{
              changeSelect.parentElement.style.display = 'none';
            }}
            key = `${{modeValue}}::stage::${{selectedStage}}::${{selectedTrack}}`;
            const stageLabel = ((mode.stage_options || []).find(function (option) {{
              return String(option.value) === String(selectedStage);
            }}) || {{}}).label || selectedStage;
            const trackLabel = ((trackOptions || []).find(function (option) {{
              return String(option.value) === String(selectedTrack);
            }}) || {{}}).label || selectedTrack;
            summaryText = `Preview shows the top ${OFFLINE_PERTURBATION_TOP_N} ranked entries for ${{stageLabel}} in ${{trackLabel}}.`;
          }} else {{
            const changeOptions = (mode.perturb_changes || []).map(function (value) {{
              return {{ value: value, label: `Perturb change ${{value}}` }};
            }});
            const selectedChange = setSelectOptions(changeSelect, changeOptions, changeSelect.value);
            const trackOptions = (((mode.tracks_by_change || {{}})[selectedChange]) || []).map(function (option) {{
              return {{ value: option.value, label: option.label || option.value }};
            }});
            const selectedTrack = setSelectOptions(trackSelect, trackOptions, trackSelect.value);
            if (stageControl) {{
              stageControl.style.display = 'none';
            }}
            if (changeSelect && changeSelect.parentElement) {{
              changeSelect.parentElement.style.display = '';
            }}
            key = `${{modeValue}}::overall::${{selectedChange}}::${{selectedTrack}}`;
            const trackLabel = ((trackOptions || []).find(function (option) {{
              return String(option.value) === String(selectedTrack);
            }}) || {{}}).label || selectedTrack;
            summaryText = `Preview shows the top ${OFFLINE_PERTURBATION_TOP_N} ranked entries for perturb change ${{selectedChange}} in ${{trackLabel}}.`;
          }}
          const state = (perturbationPayload.tables || {{}})[key];
          if (!state) {{
            plot.removeAttribute('src');
            plot.alt = `${{mode.title || modeValue}}`;
            summary.textContent = 'No perturbation result is available for this selection.';
            plotDownload.removeAttribute('href');
            tableDownload.removeAttribute('href');
            return;
          }}

          plot.src = String(state.plot_path || '');
          plot.alt = `${{mode.title || modeValue}} | ${{state.track_label || state.track || ''}}`;
          plotDownload.href = String(state.plot_download_path || '#');
          tableDownload.href = String(state.table_download_path || '#');
          summary.textContent = summaryText;
        }}

        function initializePerturbationPanels() {{
          (perturbationPayload.modes || []).forEach(function (mode) {{
            const modeValue = String(mode.value || '');
            const scopeSelect = document.getElementById(`perturb-scope-${{modeValue}}`);
            const changeSelect = document.getElementById(`perturb-change-${{modeValue}}`);
            const stageSelect = document.getElementById(`perturb-stage-${{modeValue}}`);
            const trackSelect = document.getElementById(`perturb-track-${{modeValue}}`);
            if (!scopeSelect || !changeSelect || !stageSelect || !trackSelect) {{
              return;
            }}
            scopeSelect.addEventListener('change', function () {{
              renderPerturbationPanel(modeValue);
            }});
            changeSelect.addEventListener('change', function () {{
              renderPerturbationPanel(modeValue);
            }});
            stageSelect.addEventListener('change', function () {{
              renderPerturbationPanel(modeValue);
            }});
            trackSelect.addEventListener('change', function () {{
              renderPerturbationPanel(modeValue);
            }});
            renderPerturbationPanel(modeValue);
          }});
        }}

        function formatAxisTick(value) {{
          const numeric = Number(value);
          if (!Number.isFinite(numeric)) {{
            return '';
          }}
          if (Math.abs(numeric) >= 100) {{
            return numeric.toFixed(0);
          }}
          if (Math.abs(numeric) >= 10) {{
            return numeric.toFixed(1);
          }}
          return numeric.toFixed(2).replace(/\.?0+$/, '');
        }}

        function renderLineChart(target, stageColumns, records) {{
          const validRecords = (records || []).slice(0, 8);
          if (!stageColumns.length || !validRecords.length) {{
            target.innerHTML = '<p class="empty">No dynamic marker rows are available for this selection.</p>';
            return;
          }}
          const width = 920;
          const height = 280;
          const margin = {{ top: 20, right: 20, bottom: 58, left: 76 }};
          const svg = createSvg('svg');
          svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
          const allValues = [];
          validRecords.forEach(function (record) {{
            const profile = stageColumns.map(function (column, index) {{
              const raw = Number(record[column] || 0);
              const offset = Number(record[stageColumns[0]] || 0);
              const value = raw - offset;
              allValues.push(value);
              return value;
            }});
            record.__profile = profile;
          }});
          const minValue = Math.min.apply(null, allValues);
          const maxValue = Math.max.apply(null, allValues);
          const yDomainMin = minValue === maxValue ? minValue - 1 : minValue;
          const yDomainMax = minValue === maxValue ? maxValue + 1 : maxValue;
          const xScale = scaleFactory(0, Math.max(stageColumns.length - 1, 1), margin.left, width - margin.right);
          const yScale = scaleFactory(yDomainMin, yDomainMax, height - margin.bottom, margin.top);
          const palette = ['#9d5c3f', '#26667f', '#6e7c80', '#bf8b30', '#9b3d6d', '#46836b', '#775da6', '#d46f3c'];
          const yTicks = [];
          const yTickCount = 5;
          for (let index = 0; index < yTickCount; index += 1) {{
            yTicks.push(yDomainMin + ((yDomainMax - yDomainMin) * index) / Math.max(yTickCount - 1, 1));
          }}

          yTicks.forEach(function (tickValue) {{
            const y = yScale(tickValue);
            const line = createSvg('line');
            line.setAttribute('x1', String(margin.left));
            line.setAttribute('x2', String(width - margin.right));
            line.setAttribute('y1', String(y));
            line.setAttribute('y2', String(y));
            line.setAttribute('stroke', 'rgba(31,42,46,0.10)');
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);

            const tick = createSvg('text');
            tick.setAttribute('x', String(margin.left - 10));
            tick.setAttribute('y', String(y + 4));
            tick.setAttribute('text-anchor', 'end');
            tick.setAttribute('font-size', '11');
            tick.setAttribute('fill', '#617079');
            tick.textContent = formatAxisTick(tickValue);
            svg.appendChild(tick);
          }});

          if (yDomainMin <= 0 && yDomainMax >= 0) {{
            const baseline = createSvg('line');
            baseline.setAttribute('x1', String(margin.left));
            baseline.setAttribute('x2', String(width - margin.right));
            baseline.setAttribute('y1', String(yScale(0)));
            baseline.setAttribute('y2', String(yScale(0)));
            baseline.setAttribute('stroke', 'rgba(31,42,46,0.18)');
            baseline.setAttribute('stroke-dasharray', '4 4');
            svg.appendChild(baseline);
          }}

          stageColumns.forEach(function (column, index) {{
            const x = xScale(index);
            const label = createSvg('text');
            label.setAttribute('x', String(x));
            label.setAttribute('y', String(height - 18));
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '12');
            label.setAttribute('fill', '#617079');
            label.textContent = String(column).replace(/^stage/, 'Stage ');
            svg.appendChild(label);
          }});

          const xAxisLabel = createSvg('text');
          xAxisLabel.setAttribute('x', String((margin.left + width - margin.right) / 2));
          xAxisLabel.setAttribute('y', String(height - 2));
          xAxisLabel.setAttribute('text-anchor', 'middle');
          xAxisLabel.setAttribute('font-size', '12');
          xAxisLabel.setAttribute('fill', '#617079');
          xAxisLabel.textContent = 'Stage';
          svg.appendChild(xAxisLabel);

          const yAxisLabel = createSvg('text');
          yAxisLabel.setAttribute('x', '22');
          yAxisLabel.setAttribute('y', String((margin.top + height - margin.bottom) / 2));
          yAxisLabel.setAttribute('text-anchor', 'middle');
          yAxisLabel.setAttribute('font-size', '12');
          yAxisLabel.setAttribute('fill', '#617079');
          yAxisLabel.setAttribute('transform', `rotate(-90 22 ${{(margin.top + height - margin.bottom) / 2}})`);
          yAxisLabel.textContent = 'log fold changes';
          svg.appendChild(yAxisLabel);

          validRecords.forEach(function (record, index) {{
            const polyline = createSvg('polyline');
            const color = palette[index % palette.length];
            const points = (record.__profile || []).map(function (value, pointIndex) {{
              return `${{xScale(pointIndex)}},${{yScale(Number(value))}}`;
            }}).join(' ');
            polyline.setAttribute('points', points);
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('stroke', color);
            polyline.setAttribute('stroke-width', '2.2');
            svg.appendChild(polyline);
          }});

          target.innerHTML = '';
          target.appendChild(svg);
        }}

        function renderBarChart(target, records) {{
          const validRecords = (records || []).slice(0, 12);
          if (!validRecords.length) {{
            target.innerHTML = '<p class="empty">No hierarchical marker rows are available for this selection.</p>';
            return;
          }}
          const width = 920;
          const rowHeight = 28;
          const height = Math.max(280, 72 + validRecords.length * rowHeight);
          const margin = {{ top: 20, right: 20, bottom: 46, left: 220 }};
          const svg = createSvg('svg');
          svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
          const values = validRecords.map(function (record) {{ return Number(record.log_fold_changes || 0); }});
          const maxAbs = Math.max.apply(null, values.map(function (value) {{ return Math.abs(value); }})) || 1;
          const xScale = scaleFactory(-maxAbs, maxAbs, margin.left, width - margin.right);
          const zeroX = xScale(0);
          const xTicks = [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];

          xTicks.forEach(function (tickValue) {{
            const x = xScale(tickValue);
            const line = createSvg('line');
            line.setAttribute('x1', String(x));
            line.setAttribute('x2', String(x));
            line.setAttribute('y1', String(margin.top));
            line.setAttribute('y2', String(height - margin.bottom));
            line.setAttribute('stroke', tickValue === 0 ? 'rgba(31,42,46,0.18)' : 'rgba(31,42,46,0.08)');
            line.setAttribute('stroke-width', '1');
            svg.appendChild(line);

            const tick = createSvg('text');
            tick.setAttribute('x', String(x));
            tick.setAttribute('y', String(height - margin.bottom + 18));
            tick.setAttribute('text-anchor', 'middle');
            tick.setAttribute('font-size', '11');
            tick.setAttribute('fill', '#617079');
            tick.textContent = formatAxisTick(tickValue);
            svg.appendChild(tick);
          }});

          const baseline = createSvg('line');
          baseline.setAttribute('x1', String(zeroX));
          baseline.setAttribute('x2', String(zeroX));
          baseline.setAttribute('y1', String(margin.top - 6));
          baseline.setAttribute('y2', String(height - margin.bottom));
          baseline.setAttribute('stroke', 'rgba(31,42,46,0.18)');
          svg.appendChild(baseline);

          const xAxisLabel = createSvg('text');
          xAxisLabel.setAttribute('x', String((margin.left + width - margin.right) / 2));
          xAxisLabel.setAttribute('y', String(height - 6));
          xAxisLabel.setAttribute('text-anchor', 'middle');
          xAxisLabel.setAttribute('font-size', '12');
          xAxisLabel.setAttribute('fill', '#617079');
          xAxisLabel.textContent = 'Log fold change';
          svg.appendChild(xAxisLabel);

          const yAxisLabel = createSvg('text');
          yAxisLabel.setAttribute('x', '20');
          yAxisLabel.setAttribute('y', String((margin.top + height - margin.bottom) / 2));
          yAxisLabel.setAttribute('text-anchor', 'middle');
          yAxisLabel.setAttribute('font-size', '12');
          yAxisLabel.setAttribute('fill', '#617079');
          yAxisLabel.setAttribute('transform', `rotate(-90 20 ${{(margin.top + height - margin.bottom) / 2}})`);
          yAxisLabel.textContent = 'Marker genes';
          svg.appendChild(yAxisLabel);

          validRecords.forEach(function (record, index) {{
            const y = margin.top + index * rowHeight;
            const value = Number(record.log_fold_changes || 0);
            const x = Math.min(zeroX, xScale(value));
            const widthValue = Math.abs(xScale(value) - zeroX);
            const bar = createSvg('rect');
            bar.setAttribute('x', String(x));
            bar.setAttribute('y', String(y));
            bar.setAttribute('width', String(widthValue));
            bar.setAttribute('height', '18');
            bar.setAttribute('fill', value >= 0 ? '#9d5c3f' : '#26667f');
            svg.appendChild(bar);

            const label = createSvg('text');
            label.setAttribute('x', String(margin.left - 12));
            label.setAttribute('y', String(y + 13));
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('font-size', '12');
            label.setAttribute('fill', '#203036');
            label.textContent = String(record.gene || record.Gene || `gene_${{index}}`);
            svg.appendChild(label);
          }});

          target.innerHTML = '';
          target.appendChild(svg);
        }}

        function refreshDynamicDirections() {{
          const selectedTrack = dynamicTrack.value;
          const track = (dynamicPayload.tracks || []).find(function (item) {{
            return String(item.value) === String(selectedTrack);
          }});
          const directions = (track ? track.directions : []).map(function (direction) {{
            return {{ value: direction, label: direction }};
          }});
          setSelectOptions(dynamicDirection, directions, dynamicDirection.value);
        }}

        function renderDynamicPanel() {{
          const key = `${{dynamicTrack.value}}::${{dynamicDirection.value}}`;
          const state = (dynamicPayload.tables || {{}})[key];
          if (!state) {{
            dynamicChart.innerHTML = '<p class="empty">No dynamic marker table is available for this selection.</p>';
            dynamicTable.innerHTML = '';
            dynamicDownload.removeAttribute('href');
            return;
          }}
          renderLineChart(dynamicChart, state.stage_columns || [], (state.table || {{}}).records || []);
          dynamicTable.innerHTML = renderSimpleTable(compactDynamicTable(state.table || {{}}));
          if (state.download_path) {{
            dynamicDownload.setAttribute('href', String(state.download_path));
          }} else {{
            dynamicDownload.removeAttribute('href');
          }}
        }}

        function refreshHierClusters() {{
          const stage = (hierarchicalPayload.stages || []).find(function (item) {{
            return String(item.value) === String(hierStage.value);
          }});
          const clusters = (stage ? stage.clusters : []).map(function (cluster) {{
            return {{ value: cluster.value, label: `Cluster ${{cluster.value}}` }};
          }});
          setSelectOptions(hierCluster, clusters, hierCluster.value);
        }}

        function refreshHierLevels() {{
          const stage = (hierarchicalPayload.stages || []).find(function (item) {{
            return String(item.value) === String(hierStage.value);
          }});
          const cluster = (stage && stage.clusters || []).find(function (item) {{
            return String(item.value) === String(hierCluster.value);
          }});
          const levels = (cluster ? cluster.levels : []).map(function (level) {{
            return {{ value: level, label: `Level ${{level}}` }};
          }});
          setSelectOptions(hierLevel, levels, hierLevel.value);
        }}

        function refreshHierDirections() {{
          const stage = (hierarchicalPayload.stages || []).find(function (item) {{
            return String(item.value) === String(hierStage.value);
          }});
          const cluster = (stage && stage.clusters || []).find(function (item) {{
            return String(item.value) === String(hierCluster.value);
          }});
          const directions = (((cluster || {{}}).directions_by_level || {{}})[hierLevel.value] || []).map(function (direction) {{
            return {{
              value: direction,
              label: direction === 'up' ? 'Up regulated' : direction === 'down' ? 'Down regulated' : direction,
            }};
          }});
          setSelectOptions(hierDirection, directions, hierDirection.value);
        }}

        function renderHierarchicalPanel() {{
          const key = `${{hierStage.value}}::${{hierCluster.value}}::${{hierLevel.value}}::${{hierDirection.value}}`;
          const state = (hierarchicalPayload.tables || {{}})[key];
          if (!state) {{
            hierChart.innerHTML = '<p class="empty">No hierarchical marker table is available for this selection.</p>';
            hierNote.textContent = 'Bars show signed log fold changes for the selected branch against its sibling comparison clusters.';
            hierTable.innerHTML = '';
            hierDownload.removeAttribute('href');
            return;
          }}
          renderBarChart(hierChart, (state.table || {{}}).records || []);
          const directionLabel = String(state.direction || '').toLowerCase() === 'down' ? 'down-regulated' : 'up-regulated';
          hierNote.textContent = `Bars show signed log fold changes for the top ${{directionLabel}} markers in the selected branch, compared against the sibling reference clusters.`;
          hierTable.innerHTML = renderSimpleTable(state.table || {{}});
          if (state.download_path) {{
            hierDownload.setAttribute('href', String(state.download_path));
          }} else {{
            hierDownload.removeAttribute('href');
          }}
        }}

        function renderTrackPlot(summary) {{
          const plot = summary.path_plot || {{}};
          const paths = plot.paths || [];
          if (!paths.length) {{
            trackPlot.innerHTML = '<p class="empty">No path plot was available for this track.</p>';
            return;
          }}
          const width = 880;
          const height = 260;
          const margin = {{ top: 20, right: 20, bottom: 46, left: 56 }};
          const svg = createSvg("svg");
          svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
          const maxPoints = Math.max.apply(null, paths.map(function (path) {{ return (path.values || []).length; }}));
          const allValues = [];
          paths.forEach(function (path) {{
            (path.values || []).forEach(function (value) {{ allValues.push(Number(value)); }});
          }});
          const xScale = scaleFactory(0, Math.max(maxPoints - 1, 1), margin.left, width - margin.right);
          const yScale = scaleFactory(Math.min.apply(null, allValues), Math.max.apply(null, allValues), height - margin.bottom, margin.top);
          const palette = ["#9d5c3f", "#26667f", "#6e7c80", "#bf8b30", "#9b3d6d", "#46836b"];

          (plot.stage_labels || []).forEach(function (label, index) {{
            const x = xScale(index);
            const axis = createSvg("text");
            axis.setAttribute("x", String(x));
            axis.setAttribute("y", String(height - 18));
            axis.setAttribute("text-anchor", "middle");
            axis.setAttribute("font-size", "12");
            axis.setAttribute("fill", "#617079");
            axis.textContent = String(label);
            svg.appendChild(axis);
          }});

          paths.forEach(function (path, index) {{
            const color = palette[index % palette.length];
            const polyline = createSvg("polyline");
            const points = (path.values || []).map(function (value, pointIndex) {{
              return `${{xScale(pointIndex)}},${{yScale(Number(value))}}`;
            }}).join(" ");
            polyline.setAttribute("points", points);
            polyline.setAttribute("fill", "none");
            polyline.setAttribute("stroke", color);
            polyline.setAttribute("stroke-width", "2.4");
            svg.appendChild(polyline);
          }});
          trackPlot.innerHTML = "";
          trackPlot.appendChild(svg);
        }}

        function renderPathDetails(path) {{
          if (!path) {{
            pathDetails.innerHTML = '<p class="empty">No path details are available.</p>';
            return;
          }}
          const enrichmentGroups = splitEnrichmentTables(path.enrichment);
          const enrichmentLinks = [];
          if (path.enrichment_download_path) {{
            enrichmentLinks.push(`<div class="download-links"><a href="${{String(path.enrichment_download_path)}}" download>Download enrichment details (Excel)</a></div>`);
          }}
          const enrichmentSection = enrichmentGroups.length > 1
            ? [
                `<div class="enrichment-tabs">`,
                enrichmentGroups.map(function (group, index) {{
                  return `<button type="button" class="enrichment-tab${{index === 0 ? ' is-active' : ''}}" data-enrichment-tab="${{group.key}}">${{group.label}}</button>`;
                }}).join(''),
                `</div>`,
                `<p class="chart-note">Rows are ranked by adjusted p-value, with rank 1 shown at the top of each tab.</p>`,
                enrichmentGroups.map(function (group, index) {{
                  return `<div class="enrichment-panel${{index === 0 ? ' is-active' : ''}}" data-enrichment-panel="${{group.key}}">${{renderSimpleTable(group.table)}}</div>`;
                }}).join(''),
                enrichmentLinks.join(''),
              ].join('')
            : `<p class="chart-note">Rows are ranked by adjusted p-value, with rank 1 shown at the top.</p>${{renderSimpleTable((enrichmentGroups[0] || {{ columns: [], rows: [] }}).table || {{ columns: [], rows: [] }})}}${{enrichmentLinks.join('')}}`;
          pathDetails.innerHTML = [
            `<div class="path-details-layout">`,
            `<div class="path-details-top">`,
            `<details class="track-section">`,
            `<summary>Genes</summary>`,
            `<div class="track-section-body">${{renderSimpleTable(path.genes)}}</div>`,
            `</details>`,
            `<details class="track-section">`,
            `<summary>Regulators</summary>`,
            `<div class="track-section-body">${{renderSimpleTable(path.regulators)}}</div>`,
            `</details>`,
            `</div>`,
            `<details class="track-section">`,
            `<summary>Enrichment</summary>`,
            `<div class="track-section-body">${{enrichmentSection}}</div>`,
            `</details>`,
            `</div>`
          ].join('');
          const enrichmentTabs = Array.from(pathDetails.querySelectorAll('[data-enrichment-tab]'));
          const enrichmentPanels = Array.from(pathDetails.querySelectorAll('[data-enrichment-panel]'));
          enrichmentTabs.forEach(function (button) {{
            button.addEventListener('click', function () {{
              const target = String(button.getAttribute('data-enrichment-tab') || '');
              enrichmentTabs.forEach(function (item) {{
                item.classList.toggle('is-active', item === button);
              }});
              enrichmentPanels.forEach(function (panel) {{
                panel.classList.toggle('is-active', String(panel.getAttribute('data-enrichment-panel') || '') === target);
              }});
            }});
          }});
        }}

        function openTrack(trackKey, fallbackLabel) {{
          if (!trackPayload.tracks || !trackPayload.tracks[trackKey]) {{
            return;
          }}
          const summary = trackPayload.tracks[trackKey];
          title.textContent = summary.track_label || fallbackLabel || trackKey;
          renderTrackPlot(summary);
          pathButtons.innerHTML = '';
          const paths = summary.paths || [];
          const selectedKey = summary.selected_path || (paths[0] && paths[0].path_key) || '';
          let currentPath = null;
          paths.forEach(function (path) {{
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'path-button' + (path.path_key === selectedKey ? ' active' : '');
            button.textContent = path.label || path.path_key;
            if (path.path_key === selectedKey) {{
              currentPath = path;
            }}
            button.addEventListener('click', function () {{
              pathButtons.querySelectorAll('.path-button').forEach(function (item) {{
                item.classList.remove('active');
              }});
              button.classList.add('active');
              renderPathDetails(path);
            }});
            pathButtons.appendChild(button);
          }});
          renderPathDetails(currentPath || paths[0] || null);
          modal.classList.add('visible');
          modal.setAttribute('aria-hidden', 'false');
        }}

        closeButton.addEventListener('click', function () {{
          modal.classList.remove('visible');
          modal.setAttribute('aria-hidden', 'true');
        }});
        modal.addEventListener('click', function (event) {{
          if (event.target === modal) {{
            modal.classList.remove('visible');
            modal.setAttribute('aria-hidden', 'true');
          }}
        }});

        renderLegend((graphPayload && graphPayload.nodes) || []);
        renderTemporalGraph(graphPayload || {{}});
        initializePerturbationPanels();

        setSelectOptions(dynamicTrack, (dynamicPayload.tracks || []).map(function (track) {{
          return {{ value: track.value, label: track.label || track.value }};
        }}), dynamicTrack.value);
        refreshDynamicDirections();
        dynamicTrack.addEventListener('change', function () {{
          refreshDynamicDirections();
          renderDynamicPanel();
        }});
        dynamicDirection.addEventListener('change', renderDynamicPanel);
        renderDynamicPanel();

        setSelectOptions(hierStage, (hierarchicalPayload.stages || []).map(function (stage) {{
          return {{ value: stage.value, label: `Stage ${{stage.value}}` }};
        }}), hierStage.value);
        refreshHierClusters();
        refreshHierLevels();
        hierStage.addEventListener('change', function () {{
          refreshHierClusters();
          refreshHierLevels();
          refreshHierDirections();
          renderHierarchicalPanel();
        }});
        hierCluster.addEventListener('change', function () {{
          refreshHierLevels();
          refreshHierDirections();
          renderHierarchicalPanel();
        }});
        hierLevel.addEventListener('change', function () {{
          refreshHierDirections();
          renderHierarchicalPanel();
        }});
        hierDirection.addEventListener('change', renderHierarchicalPanel);
        refreshHierDirections();
        renderHierarchicalPanel();
      }})();
    </script>
  </body>
</html>
"""


def _zip_bundle(bundle_dir: Path, zip_path: Path, *, archive_root_name: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=str(Path(archive_root_name) / path.relative_to(bundle_dir)))


def export_analysis_share_bundle(
    analysis_dir: str | Path,
    *,
    force: bool = False,
    analysis_label: str | None = None,
) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).expanduser().resolve()
    bundle_dir, manifest_path, _legacy_zip_path = _bundle_paths(analysis_path)
    expected_fingerprint = _source_fingerprint(analysis_path)

    if not force:
        existing = _existing_bundle_manifest(analysis_path)
        if existing and existing.get("source_fingerprint") == expected_fingerprint:
            return existing

    visual_manifest_path = analysis_path / "visualizations" / "manifest.json"
    if visual_manifest_path.exists():
        visual_manifest = json.loads(visual_manifest_path.read_text(encoding="utf-8"))
    else:
        visual_manifest = generate_analysis_visualizations(analysis_path)

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    _cleanup_old_share_bundle_archives(analysis_path)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    entries = _collect_entries(visual_manifest, bundle_dir=bundle_dir, analysis_dir=analysis_path)
    adata, _dataset_path, _attribute_path = _read_analysis_adata(
        analysis_path,
        uns_keys=INTERACTIVE_UNS_KEYS,
    )
    generated_dt = datetime.now(timezone.utc)
    generated_at = generated_dt.isoformat()
    bundle_title = analysis_label or analysis_path.name
    download_prefix = _safe_name(bundle_title)
    _override_stage_umap_entry(entries, bundle_dir=bundle_dir, adata=adata)
    temporal_graph = _build_temporal_graph_payload(analysis_path, adata)
    track_summaries = _build_track_summaries(
        analysis_path,
        temporal_graph,
        bundle_dir=bundle_dir,
        download_prefix=download_prefix,
    )
    perturbation_bundle = _build_perturbation_bundle(
        analysis_path,
        adata,
        bundle_dir=bundle_dir,
        download_prefix=download_prefix,
    )
    dynamic_markers = _build_dynamic_marker_bundle(
        adata,
        bundle_dir=bundle_dir,
        download_prefix=download_prefix,
    )
    hierarchical_markers = _build_hierarchical_marker_bundle(
        adata,
        bundle_dir=bundle_dir,
        download_prefix=download_prefix,
    )

    archive_base_name = _share_bundle_base_name(analysis_path, bundle_title, generated_dt)
    archive_root_name = archive_base_name
    zip_path = analysis_path / f"{archive_base_name}.zip"
    html_text = _render_report_html(
        bundle_title=bundle_title,
        generated_at=generated_at,
        entries=entries,
        perturbation_bundle=perturbation_bundle,
        temporal_graph=temporal_graph,
        track_summaries=track_summaries,
        dynamic_markers=dynamic_markers,
        hierarchical_markers=hierarchical_markers,
    )
    index_path = bundle_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")

    manifest = {
        "version": SHARE_BUNDLE_VERSION,
        "analysis_dir": str(analysis_path),
        "analysis_label": bundle_title,
        "archive_base_name": archive_base_name,
        "generated_at": generated_at,
        "source_fingerprint": expected_fingerprint,
        "bundle_dir": str(bundle_dir),
        "index_path": str(index_path),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "entries": entries,
        "counts": {
            "entries": len(entries),
            "tracks": len(track_summaries.get("tracks", {})),
            "track_errors": len(track_summaries.get("errors", {})),
        },
        "temporal_graph": {
            "available": temporal_graph is not None,
            "node_count": int((temporal_graph or {}).get("node_count", 0)),
            "edge_count": int((temporal_graph or {}).get("edge_count", 0)),
        },
        "track_summaries": {
            "available": bool(track_summaries.get("tracks")),
            "track_keys": sorted(track_summaries.get("tracks", {}).keys()),
            "errors": track_summaries.get("errors", {}),
        },
        "dynamic_markers": {
            "available": bool(dynamic_markers.get("available")),
            "track_count": len(dynamic_markers.get("tracks", [])),
        },
        "hierarchical_markers": {
            "available": bool(hierarchical_markers.get("available")),
            "stage_count": len(hierarchical_markers.get("stages", [])),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _zip_bundle(bundle_dir, zip_path, archive_root_name=archive_root_name)
    return manifest
