from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


DEFAULT_CATEGORICAL_COLORS: Tuple[str, ...] = (
    "#4DBBD5",
    "#F39B7F",
    "#00A087",
    "#E64B35",
    "#3C5488",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
)


def _prepare_scanpy():
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    import scanpy as sc

    return sc


def _sorted_stage_labels(values: Iterable[Any]) -> list[Any]:
    vals = list(values)
    try:
        return sorted(vals, key=lambda x: float(str(x)))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda x: str(x))


def build_palette(categories: Sequence[Any], preset: Optional[Mapping[Any, str]] = None) -> Dict[str, str]:
    cats = [str(v) for v in categories]
    palette: Dict[str, str] = {}
    if preset:
        palette.update({str(k): str(v) for k, v in preset.items()})
    missing = [cat for cat in cats if cat not in palette]
    for i, cat in enumerate(missing):
        palette[cat] = DEFAULT_CATEGORICAL_COLORS[i % len(DEFAULT_CATEGORICAL_COLORS)]
    return palette


def _safe_numeric_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def _resolve_key_by_string(mapping: Mapping[Any, Any], key: Any, what: str) -> Any:
    if key in mapping:
        return key
    key_s = str(key)
    if key_s in mapping:
        return key_s
    for existing in mapping.keys():
        if str(existing) == key_s:
            return existing
    raise KeyError(f"{what}={key!r} not found. Available: {list(mapping.keys())}")


def _wrap_ranked_bar_label(value: Any, *, max_line_length: int = 34, max_lines: int = 2) -> str:
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
    return "\n".join(wrapped[:max_lines])


def _dominant_label(series: pd.Series, fallback: str = "NA") -> str:
    if series.empty:
        return fallback
    mode_vals = series.astype(str).mode(dropna=True)
    if mode_vals.empty:
        return fallback
    return str(mode_vals.iloc[0])


def _abbreviate_track_label(cell_type: str) -> str:
    ct = str(cell_type)
    preset = {
        "FibroblastAdventitial": "FibAdv",
        "FibroblastAlveolar": "FibAlv",
        "FibroblastFibrotic": "FibFib",
        "FibroblastAirway": "FibAir",
        "LymphEndo": "LymEndo",
        "PericyteAlveolar": "PeriAlv",
        "SMC": "SMC",
        "SMCAirway": "SMCAir",
        "VEaerocyte": "VEAero",
        "VEarterial": "VEArt",
        "VEcapillary": "VECap",
        "VEvenous": "VEVein",
    }
    if ct in preset:
        return preset[ct]
    cleaned = "".join(ch for ch in ct if ch.isalnum())
    return cleaned[:10] if cleaned else "Track"


def plot_stage_umap_grid(
    adata,
    *,
    stage_key: str = "stage",
    color_keys: Sequence[str] = ("ident", "leiden"),
    stage_order: Optional[Sequence[Any]] = None,
    palette: Optional[Mapping[Any, str]] = None,
    size: float = 8.0,
    dpi: int = 100,
    figsize_per_panel: Tuple[float, float] = (6.0, 3.6),
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot manuscript-style stage-wise UMAP panels (e.g., Fig. 2/6/7 style).
    """
    sc = _prepare_scanpy()
    if "X_umap" not in adata.obsm:
        if "umap" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["umap"]
        else:
            raise KeyError("UMAP embedding missing. Expected adata.obsm['X_umap'] or adata.obsm['umap'].")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")

    colors = list(color_keys)
    if not colors:
        raise ValueError("color_keys must contain at least one obs column.")
    for key in colors:
        if key not in adata.obs:
            raise KeyError(f"Missing color key in adata.obs: {key}")

    stages = list(stage_order) if stage_order is not None else _sorted_stage_labels(adata.obs[stage_key].unique().tolist())
    n_rows = len(stages)
    n_cols = len(colors)
    categorical_palettes: Dict[str, Dict[str, str]] = {}
    for color_key in colors:
        series = adata.obs[color_key]
        if series.dtype.name == "category" or series.dtype == object:
            categories = sorted(series.astype(str).unique().tolist(), key=_safe_numeric_sort_key)
            categorical_palettes[color_key] = build_palette(categories, preset=palette)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        dpi=dpi,
        squeeze=False,
    )

    for r, stage in enumerate(stages):
        stage_mask = adata.obs[stage_key].astype(str) == str(stage)
        stage_adata = adata[stage_mask].to_memory() if adata.isbacked else adata[stage_mask].copy()
        if stage_adata.n_obs == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
                axes[r, c].set_title(f"{stage_key}={stage} (empty)")
            continue

        for c, color_key in enumerate(colors):
            ax = axes[r, c]
            sc.pl.umap(
                stage_adata,
                color=color_key,
                ax=ax,
                show=False,
                size=size,
                title=f"{stage_key}={stage} | {color_key}",
                palette=categorical_palettes.get(color_key),
            )

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_stage_cell_type_composition(
    adata,
    *,
    cell_type_key: str = "ident",
    stage_key: str = "stage",
    min_label_fraction: float = 0.04,
    stage_descending: bool = True,
    palette: Optional[Mapping[Any, str]] = None,
    figsize: Tuple[float, float] = (12.0, 4.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot stacked horizontal cell-type composition by stage (Fig. 2/7 style).
    """
    if cell_type_key not in adata.obs:
        raise KeyError(f"Missing cell type key in adata.obs: {cell_type_key}")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")

    ctab = pd.crosstab(
        adata.obs[stage_key].astype(str),
        adata.obs[cell_type_key].astype(str),
        normalize="index",
    )
    stage_order = _sorted_stage_labels(ctab.index.tolist())
    if stage_descending:
        stage_order = stage_order[::-1]
    ctab = ctab.loc[[str(v) for v in stage_order]]
    ctab = ctab[sorted(ctab.columns.tolist())]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    palette_map = build_palette(ctab.columns.tolist(), preset=palette)
    left = np.zeros(len(ctab), dtype=float)
    y_pos = np.arange(len(ctab))
    for col in ctab.columns:
        vals = ctab[col].to_numpy(dtype=float)
        bars = ax.barh(
            y=y_pos,
            width=vals,
            left=left,
            color=palette_map[str(col)],
            edgecolor="white",
            linewidth=0.4,
            label=str(col),
        )
        labels = [f"{v*100:.1f}%" if v >= min_label_fraction else "" for v in vals]
        ax.bar_label(bars, labels=labels, label_type="center", fontsize=8)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ctab.index.tolist())
    ax.set_xlabel("Fraction")
    ax.set_ylabel(stage_key)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        title=cell_type_key,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, ctab


def plot_marker_dotplot(
    adata,
    marker_map: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    groupby: str = "ident",
    layer: Optional[str] = None,
    standard_scale: Optional[str] = "var",
    dpi: int = 220,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Dot plot for marker programs (Fig. 2b/7b style).
    """
    sc = _prepare_scanpy()
    if groupby not in adata.obs:
        raise KeyError(f"Missing groupby key in adata.obs: {groupby}")
    sc.set_figure_params(dpi=dpi, dpi_save=dpi)
    dot = sc.pl.dotplot(
        adata,
        var_names=marker_map,
        groupby=groupby,
        layer=layer,
        standard_scale=standard_scale,
        return_fig=True,
        show=False,
    )
    if save:
        dot.savefig(save, dpi=dpi)
    if show:
        dot.show()
    return dot


def plot_dynamic_trajectory_graph(
    edges: pd.DataFrame | Sequence[Tuple[Any, Any]],
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    stage_map: Optional[Mapping[Any, Any]] = None,
    node_color_map: Optional[Mapping[Any, str]] = None,
    node_size: int = 600,
    edge_alpha: float = 0.5,
    edge_width: float = 1.8,
    with_labels: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.0),
    dpi: int = 220,
    seed: int = 8848,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a directed trajectory graph (Fig. 3a/7c style).
    """
    if isinstance(edges, pd.DataFrame):
        if source_col not in edges.columns or target_col not in edges.columns:
            raise KeyError(f"edges DataFrame must include columns: {source_col}, {target_col}")
        edge_df = edges[[source_col, target_col] + ([weight_col] if weight_col and weight_col in edges.columns else [])].copy()
    else:
        edge_df = pd.DataFrame(edges, columns=[source_col, target_col])

    graph = nx.DiGraph()
    for _, row in edge_df.iterrows():
        src, dst = row[source_col], row[target_col]
        if weight_col and weight_col in edge_df.columns:
            graph.add_edge(src, dst, weight=float(row[weight_col]))
        else:
            graph.add_edge(src, dst, weight=1.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if stage_map:
        groups: Dict[str, list[Any]] = {}
        for node in graph.nodes():
            stage = str(stage_map.get(node, "NA"))
            groups.setdefault(stage, []).append(node)
        stage_keys = _sorted_stage_labels(groups.keys())
        pos: Dict[Any, Tuple[float, float]] = {}
        for x_idx, stage in enumerate(stage_keys):
            nodes = sorted(groups[str(stage)], key=lambda v: str(v))
            if len(nodes) == 1:
                y_vals = np.array([0.0])
            else:
                y_vals = np.linspace(-1.0, 1.0, len(nodes))
            for node, y in zip(nodes, y_vals):
                pos[node] = (float(x_idx), float(y))
    else:
        pos = nx.spring_layout(graph, seed=seed)

    node_colors = []
    fallback = list(DEFAULT_CATEGORICAL_COLORS)
    for i, node in enumerate(graph.nodes()):
        if node_color_map and node in node_color_map:
            node_colors.append(str(node_color_map[node]))
        else:
            node_colors.append(fallback[i % len(fallback)])

    widths = []
    for _, _, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        widths.append(max(0.6, edge_width * w))

    nx.draw_networkx_nodes(graph, pos=pos, node_color=node_colors, node_size=node_size, ax=ax, linewidths=0.5, edgecolors="black")
    nx.draw_networkx_edges(graph, pos=pos, width=widths, alpha=edge_alpha, arrows=True, arrowstyle="-|>", arrowsize=16, ax=ax)
    if with_labels:
        nx.draw_networkx_labels(graph, pos=pos, font_size=8, ax=ax)

    ax.set_axis_off()
    ax.set_title("Dynamic trajectory graph")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _get_progression_markers(adata) -> Mapping[str, Any]:
    if "progressionMarkers" not in adata.uns:
        raise KeyError("Missing adata.uns['progressionMarkers']. Run dynamic marker discovery first.")
    markers = adata.uns["progressionMarkers"]
    if not isinstance(markers, Mapping):
        raise TypeError("adata.uns['progressionMarkers'] must be a mapping.")
    return markers


def _progression_stage_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if str(c).startswith("stage")]
    if not cols:
        raise KeyError("No stage columns found in progression markers table.")
    return [str(c) for c in sorted(cols, key=_safe_numeric_sort_key)]


def extract_dynamic_marker_table(
    adata,
    *,
    track: str,
    direction: str = "increasing",
    top_n: Optional[int] = None,
    sort_by: str = "rank",
) -> pd.DataFrame:
    """
    Extract one track-direction dynamic marker table from `adata.uns['progressionMarkers']`.
    """
    progression = _get_progression_markers(adata)
    track_key = _resolve_key_by_string(progression, track, "track")
    track_bucket = progression[track_key]
    if not isinstance(track_bucket, Mapping):
        raise TypeError(f"progressionMarkers[{track_key!r}] must be a mapping.")

    direction_key = _resolve_key_by_string(track_bucket, direction, "direction")
    raw = track_bucket[direction_key]
    table = pd.DataFrame.from_dict(raw).copy()
    if table.empty:
        return table

    if "gene" not in table.columns:
        table = table.rename_axis("gene").reset_index()
    table["track"] = str(track_key)
    table["direction"] = str(direction_key)

    stage_cols = _progression_stage_columns(table)
    for col in stage_cols + [sort_by]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=True, na_position="last")
    if top_n is not None and top_n > 0:
        table = table.head(int(top_n)).copy()
    table = table.reset_index(drop=True)
    return table


def _normalize_stage_profile(values: np.ndarray, normalize_to_stage0: bool) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1)
    if normalize_to_stage0 and out.size > 0:
        out = out - out[0]
    return out


def plot_dynamic_marker_trajectories(
    adata,
    *,
    tracks: Sequence[str] | str,
    directions: Sequence[str] = ("increasing", "decreasing"),
    top_n: int = 10,
    normalize_to_stage0: bool = True,
    stage_label_map: Optional[Mapping[Any, str]] = None,
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (5.6, 4.2),
    linewidth: float = 1.6,
    alpha: float = 0.9,
    legend_max_genes: int = 28,
    dpi: int = 220,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot top dynamic-marker trajectories per track (Fig. 3d / Fig. 4-like).
    """
    track_list = [tracks] if isinstance(tracks, str) else list(tracks)
    if not track_list:
        raise ValueError("tracks must be non-empty.")
    direction_list = [str(d) for d in directions if str(d)]
    if not direction_list:
        raise ValueError("directions must be non-empty.")

    n_panels = len(track_list)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=dpi,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    records: list[dict[str, Any]] = []

    for i, track in enumerate(track_list):
        ax = flat_axes[i]
        plotted_labels: list[str] = []
        line_count = 0
        x_ticks = None
        x_labels: list[str] = []

        for d_idx, direction in enumerate(direction_list):
            table = extract_dynamic_marker_table(
                adata,
                track=str(track),
                direction=str(direction),
                top_n=top_n,
                sort_by="rank",
            )
            if table.empty:
                continue
            stage_cols = _progression_stage_columns(table)
            x = np.arange(len(stage_cols), dtype=float)
            x_ticks = x
            x_labels = []
            for col in stage_cols:
                idx_s = str(col).replace("stage", "")
                key_candidates = [col, idx_s]
                mapped = None
                if stage_label_map:
                    for key in key_candidates:
                        if key in stage_label_map:
                            mapped = stage_label_map[key]
                            break
                        try:
                            as_int = int(key)
                            if as_int in stage_label_map:
                                mapped = stage_label_map[as_int]
                                break
                        except (TypeError, ValueError):
                            pass
                x_labels.append(str(mapped) if mapped is not None else str(col))

            color_cycle = plt.cm.tab20(np.linspace(0, 1, max(1, len(table))))
            linestyle = "-" if str(direction).lower().startswith("inc") else "--"
            for r_idx, row in table.iterrows():
                gene = str(row.get("gene", f"gene_{r_idx}"))
                profile = _normalize_stage_profile(row[stage_cols].to_numpy(dtype=float), normalize_to_stage0=normalize_to_stage0)
                label = f"{gene} ({direction[0]})"
                ax.plot(x, profile, linewidth=linewidth, alpha=alpha, linestyle=linestyle, color=color_cycle[r_idx % len(color_cycle)], label=label)
                plotted_labels.append(label)
                line_count += 1
                for j, stage_col in enumerate(stage_cols):
                    records.append(
                        {
                            "track": str(track),
                            "direction": str(direction),
                            "gene": gene,
                            "stage": str(stage_col),
                            "x": int(j),
                            "value": float(profile[j]),
                        }
                    )

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
        if normalize_to_stage0:
            ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_ylabel("Expression change vs stage0")
        else:
            ax.set_ylabel("Expression")
        ax.set_xlabel("Stage")
        ax.set_title(f"Track {track}")
        ax.grid(axis="y", alpha=0.2)
        if 0 < line_count <= legend_max_genes:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
        elif line_count > legend_max_genes:
            ax.text(
                0.99,
                0.99,
                f"{line_count} lines (legend hidden)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="grey",
            )

    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, pd.DataFrame.from_records(records)


def plot_dynamic_marker_heatmap(
    adata,
    *,
    tracks: Optional[Sequence[str]] = None,
    directions: Sequence[str] = ("increasing", "decreasing"),
    top_n: int = 8,
    per_track_total: bool = False,
    normalize_to_stage0: bool = True,
    protein_coding_only: bool = False,
    require_genes_in_var: bool = False,
    noncoding_prefixes: Sequence[str] = ("AC", "AL", "AP", "LINC"),
    exclude_dot_genes: bool = False,
    row_label_mode: str = "full",
    row_zscore: bool = False,
    cell_level: bool = False,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    cell_groupby: str = "stage",
    restrict_to_track_cells: bool = True,
    scale_layer: bool = False,
    stage_label_map: Optional[Mapping[Any, str]] = None,
    show_track_separators: bool = True,
    separator_color: str = "#BFBFBF",
    separator_linewidth: float = 0.8,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (11.0, 9.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Heatmap panel of top dynamic markers across tracks (Fig. 4a-style summary view).

    Parameters
    ----------
    top_n:
        If ``per_track_total=False`` (default), this is applied per track *and* per
        direction. If ``per_track_total=True``, this is the total marker budget per
        track across all selected directions (use 3 or 5 for Fig. 4A-like compact panels).
    row_label_mode:
        One of ``full`` (track|direction|gene), ``track_gene`` (track|gene), or ``gene``.
    cell_level:
        If ``True``, plot selected dynamic-marker genes across individual cells
        (template-style) rather than stage-aggregated progression profiles.
    cell_groupby:
        Grouping for cell-level heatmap. One of ``stage`` (default),
        ``stage_cluster`` or ``cluster``.
    row_zscore / scale_layer:
        In ``cell_level=True`` mode, expression uses z-score normalization when either
        ``row_zscore=True`` or ``scale_layer=True``. Z-scores are computed per gene
        across all cells in ``adata`` (then applied to the plotted subset).
    """
    progression = _get_progression_markers(adata)
    chosen_tracks = list(tracks) if tracks is not None else _sorted_stage_labels(list(progression.keys()))
    if not chosen_tracks:
        raise ValueError("No tracks available for plotting.")
    direction_list = [str(d) for d in directions if str(d)]
    if not direction_list:
        raise ValueError("directions must be non-empty.")

    mat_rows: list[np.ndarray] = []
    row_labels: list[str] = []
    column_labels: list[str] = []
    row_tracks: list[str] = []
    selected_gene_rows: list[tuple[str, str, str]] = []
    label_mode = str(row_label_mode).strip().lower()
    if label_mode not in {"full", "track_gene", "gene"}:
        raise ValueError("row_label_mode must be one of: full, track_gene, gene")
    prefixes = tuple(str(p).upper() for p in noncoding_prefixes if str(p))
    var_gene_set = set(str(g) for g in adata.var_names) if require_genes_in_var else None

    for track in chosen_tracks:
        if per_track_total:
            n_dirs = len(direction_list)
            if top_n is not None and top_n > 0:
                base = int(top_n) // n_dirs
                rem = int(top_n) % n_dirs
                quotas = {d: base + (1 if idx < rem else 0) for idx, d in enumerate(direction_list)}
            else:
                quotas = {d: top_n for d in direction_list}
            direction_tables: list[pd.DataFrame] = []
            for direction in direction_list:
                quota = quotas.get(direction, 0)
                if quota is not None and quota <= 0:
                    continue
                table = extract_dynamic_marker_table(
                    adata,
                    track=str(track),
                    direction=str(direction),
                    top_n=quota,
                    sort_by="rank",
                )
                if table.empty:
                    continue
                table = table.copy()
                table["direction"] = str(direction)
                direction_tables.append(table)
            if not direction_tables:
                continue
            selected_tables = direction_tables
        else:
            selected_tables = []
            for direction in direction_list:
                table = extract_dynamic_marker_table(
                    adata,
                    track=str(track),
                    direction=str(direction),
                    top_n=top_n,
                    sort_by="rank",
                )
                if table.empty:
                    continue
                selected_tables.append(table)

        for table in selected_tables:
            if table.empty:
                continue
            direction = str(table.get("direction").iloc[0]) if "direction" in table.columns else ""
            stage_cols = _progression_stage_columns(table)
            if not column_labels:
                for col in stage_cols:
                    idx_s = str(col).replace("stage", "")
                    mapped = None
                    if stage_label_map:
                        if col in stage_label_map:
                            mapped = stage_label_map[col]
                        elif idx_s in stage_label_map:
                            mapped = stage_label_map[idx_s]
                        else:
                            try:
                                idx_i = int(idx_s)
                                if idx_i in stage_label_map:
                                    mapped = stage_label_map[idx_i]
                            except (TypeError, ValueError):
                                pass
                    column_labels.append(str(mapped) if mapped is not None else str(col))
            for _, row in table.iterrows():
                gene = str(row.get("gene", "gene"))
                gene_upper = gene.upper()
                if var_gene_set is not None and gene not in var_gene_set:
                    continue
                if protein_coding_only:
                    if prefixes and any(gene_upper.startswith(pref) for pref in prefixes):
                        continue
                    if exclude_dot_genes and "." in gene:
                        continue
                profile = _normalize_stage_profile(row[stage_cols].to_numpy(dtype=float), normalize_to_stage0=normalize_to_stage0)
                mat_rows.append(profile)
                row_tracks.append(str(track))
                selected_gene_rows.append((str(track), str(direction), gene))
                if label_mode == "gene":
                    row_labels.append(gene)
                elif label_mode == "track_gene":
                    row_labels.append(f"{track} | {gene}")
                else:
                    if direction:
                        dir_tag = "↑" if direction.lower().startswith("inc") else ("↓" if direction.lower().startswith("dec") else direction[0])
                        row_labels.append(f"{track} | {dir_tag} | {gene}")
                    else:
                        row_labels.append(f"{track} | {gene}")

    if not mat_rows:
        raise ValueError("No dynamic marker rows were extracted from the selected tracks/directions.")

    if cell_level:
        if ax is not None:
            raise ValueError("`ax` is not supported for plot_dynamic_marker_heatmap when using sc.pl.heatmap.")
        sc = _prepare_scanpy()

        marker_genes: list[str] = []
        seen_genes: set[str] = set()
        for _, _, gene in selected_gene_rows:
            if gene not in seen_genes:
                seen_genes.add(gene)
                marker_genes.append(gene)
        if not marker_genes:
            raise ValueError("No marker genes available after filtering for cell-level heatmap.")

        obs_mask = np.ones(adata.n_obs, dtype=bool)
        if stage_key in adata.obs.columns and column_labels:
            valid_stages = set(str(col).replace("stage", "") for col in column_labels)
            obs_mask &= adata.obs[stage_key].astype(str).isin(valid_stages).to_numpy(dtype=bool)

        if restrict_to_track_cells and stage_key in adata.obs.columns and cluster_key in adata.obs.columns:
            allowed_pairs: set[tuple[str, str]] = set()
            for track in chosen_tracks:
                parts = str(track).split("-")
                for s_idx, token in enumerate(parts):
                    for cluster_token in str(token).split("n"):
                        cluster_id = str(cluster_token).strip()
                        if cluster_id:
                            allowed_pairs.add((str(s_idx), cluster_id))
            if allowed_pairs:
                stages = adata.obs[stage_key].astype(str).to_numpy()
                clusters = adata.obs[cluster_key].astype(str).to_numpy()
                pair_mask = np.fromiter(
                    ((stages[i], clusters[i]) in allowed_pairs for i in range(adata.n_obs)),
                    dtype=bool,
                    count=adata.n_obs,
                )
                obs_mask &= pair_mask

        cell_adata = adata[obs_mask, marker_genes].to_memory() if adata.isbacked else adata[obs_mask, marker_genes].copy()
        if cell_adata.n_obs == 0:
            raise ValueError("No cells left after filtering for cell-level dynamic heatmap.")

        groupby_mode = str(cell_groupby).strip().lower()
        if groupby_mode not in {"stage", "stage_cluster", "cluster"}:
            raise ValueError("cell_groupby must be one of: stage, stage_cluster, cluster")

        has_stage = stage_key in cell_adata.obs.columns
        has_cluster = cluster_key in cell_adata.obs.columns

        if has_stage:
            stage_vals = cell_adata.obs[stage_key].astype(str)
            stage_order = [str(v) for v in _sorted_stage_labels(stage_vals.unique().tolist())]
            stage_labels = [f"stage {s}" for s in stage_order]
            stage_label_values = stage_vals.map(lambda s: f"stage {s}")
            cell_adata.obs["_stage_label"] = pd.Categorical(
                stage_label_values,
                categories=stage_labels,
                ordered=True,
            )
        if has_cluster:
            cluster_vals = cell_adata.obs[cluster_key].astype(str)
            cluster_order = [str(v) for v in _sorted_stage_labels(cluster_vals.unique().tolist())]
            cell_adata.obs[cluster_key] = pd.Categorical(cluster_vals, categories=cluster_order, ordered=True)

        if groupby_mode == "stage_cluster" and has_stage and has_cluster:
            stage_vals = cell_adata.obs[stage_key].astype(str)
            cluster_vals = cell_adata.obs[cluster_key].astype(str)
            group_vals = stage_vals + "|" + cluster_vals
            order_df = pd.DataFrame({"stage": stage_vals, "cluster": cluster_vals, "group": group_vals}).drop_duplicates()
            order_df["stage_sort"] = order_df["stage"].apply(_safe_numeric_sort_key)
            order_df["cluster_sort"] = order_df["cluster"].apply(_safe_numeric_sort_key)
            order_df = order_df.sort_values(["stage_sort", "cluster_sort"])
            group_order = order_df["group"].tolist()
            cell_adata.obs["_dynamic_group"] = pd.Categorical(group_vals, categories=group_order, ordered=True)
            groupby_key: str | list[str] = "_dynamic_group"
        elif groupby_mode == "cluster" and has_cluster:
            groupby_key = cluster_key
        elif has_stage:
            groupby_key = "_stage_label"
        elif has_cluster:
            groupby_key = cluster_key
        else:
            cell_adata.obs["_dynamic_group"] = pd.Categorical(
                [str(i) for i in range(cell_adata.n_obs)],
                categories=[str(i) for i in range(cell_adata.n_obs)],
                ordered=True,
            )
            groupby_key = "_dynamic_group"

        use_scaled = bool(scale_layer or row_zscore)
        layer_name: Optional[str] = None
        if use_scaled:
            full_matrix = adata[:, marker_genes].X
            if hasattr(full_matrix, "toarray"):
                full_arr = full_matrix.toarray()
            else:
                full_arr = np.asarray(full_matrix)
            if full_arr.ndim != 2:
                full_arr = np.asarray(full_arr).reshape(adata.n_obs, len(marker_genes))
            gene_means = full_arr.mean(axis=0, keepdims=True)
            gene_stds = full_arr.std(axis=0, keepdims=True)
            gene_stds[gene_stds == 0.0] = 1.0

            subset_matrix = cell_adata.X
            if hasattr(subset_matrix, "toarray"):
                subset_arr = subset_matrix.toarray()
            else:
                subset_arr = np.asarray(subset_matrix)
            if subset_arr.ndim != 2:
                subset_arr = np.asarray(subset_arr).reshape(cell_adata.n_obs, len(marker_genes))
            cell_adata.layers["scaled"] = (subset_arr - gene_means) / gene_stds
            layer_name = "scaled"

        hm_axes = sc.pl.heatmap(
            cell_adata,
            var_names=marker_genes,
            groupby=groupby_key,
            layer=layer_name,
            cmap=cmap,
            dendrogram=False,
            swap_axes=True,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            show=False,
        )
        heatmap_ax = hm_axes.get("heatmap_ax") if isinstance(hm_axes, dict) else hm_axes
        groupby_ax = hm_axes.get("groupby_ax") if isinstance(hm_axes, dict) else None
        fig = heatmap_ax.figure if heatmap_ax is not None else (groupby_ax.figure if groupby_ax is not None else plt.gcf())

        if heatmap_ax is not None:
            heatmap_ax.set_title("Dynamic marker heatmap (cell-level)")
        if groupby_ax is not None:
            if groupby_key == "_dynamic_group":
                groupby_ax.set_ylabel(f"{stage_key}|{cluster_key}")
            elif groupby_key == "_stage_label":
                groupby_ax.set_ylabel("stage")
            else:
                groupby_ax.set_ylabel(str(groupby_key))

        cbar_label = "z-score" if use_scaled else "Expression"
        aux_axes = [a for a in fig.axes if a is not heatmap_ax and a is not groupby_ax]
        if aux_axes:
            aux_axes[0].set_ylabel(cbar_label)
        fig.set_dpi(dpi)

        cell_matrix = cell_adata.layers[layer_name] if layer_name else cell_adata.X
        if hasattr(cell_matrix, "toarray"):
            cell_matrix_2d = cell_matrix.toarray()
        else:
            cell_matrix_2d = np.asarray(cell_matrix)
        matrix = pd.DataFrame(
            cell_matrix_2d,
            index=cell_adata.obs_names.astype(str),
            columns=cell_adata.var_names.astype(str),
        )
        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig, heatmap_ax, matrix

    matrix = pd.DataFrame(np.vstack(mat_rows), index=row_labels, columns=column_labels)
    if row_zscore:
        vals = matrix.to_numpy(dtype=float)
        means = vals.mean(axis=1, keepdims=True)
        stds = vals.std(axis=1, keepdims=True)
        stds[stds == 0.0] = 1.0
        matrix = pd.DataFrame((vals - means) / stds, index=matrix.index, columns=matrix.columns)
    if ax is not None:
        raise ValueError("`ax` is not supported for plot_dynamic_marker_heatmap when using sc.pl.heatmap.")

    sc = _prepare_scanpy()
    from anndata import AnnData

    # Make row labels unique so each marker row is rendered as its own group in scanpy.
    unique_row_labels: list[str] = []
    seen_row_labels: dict[str, int] = {}
    for label in matrix.index.astype(str).tolist():
        count = seen_row_labels.get(label, 0) + 1
        seen_row_labels[label] = count
        unique_row_labels.append(label if count == 1 else f"{label} [{count}]")

    plot_obs = pd.DataFrame({"_dynamic_marker_row": unique_row_labels}, index=[f"row_{i}" for i in range(len(unique_row_labels))])
    plot_obs["_dynamic_marker_row"] = pd.Categorical(
        plot_obs["_dynamic_marker_row"],
        categories=unique_row_labels,
        ordered=True,
    )
    plot_var = pd.DataFrame(index=[str(c) for c in matrix.columns])
    plot_adata = AnnData(X=matrix.to_numpy(dtype=float), obs=plot_obs, var=plot_var, dtype=float)

    hm_axes = sc.pl.heatmap(
        plot_adata,
        var_names=list(plot_adata.var_names),
        groupby="_dynamic_marker_row",
        cmap=cmap,
        dendrogram=False,
        swap_axes=False,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        show=False,
    )

    heatmap_ax = hm_axes.get("heatmap_ax") if isinstance(hm_axes, dict) else hm_axes
    groupby_ax = hm_axes.get("groupby_ax") if isinstance(hm_axes, dict) else None
    fig = heatmap_ax.figure if heatmap_ax is not None else (groupby_ax.figure if groupby_ax is not None else plt.gcf())

    if show_track_separators and len(row_tracks) > 1:
        boundaries = [i for i in range(1, len(row_tracks)) if row_tracks[i] != row_tracks[i - 1]]
        for boundary in boundaries:
            y = boundary - 0.5
            if heatmap_ax is not None:
                heatmap_ax.hlines(y, *heatmap_ax.get_xlim(), colors=separator_color, linewidth=separator_linewidth)
            if groupby_ax is not None:
                groupby_ax.hlines(y, *groupby_ax.get_xlim(), colors=separator_color, linewidth=separator_linewidth)

    if heatmap_ax is not None:
        heatmap_ax.set_xlabel("Stage")
        heatmap_ax.set_title("Dynamic marker heatmap")
    if groupby_ax is not None:
        if label_mode == "gene":
            groupby_ax.set_ylabel("Gene")
        elif label_mode == "track_gene":
            groupby_ax.set_ylabel("Track | Gene")
        else:
            groupby_ax.set_ylabel("Track | Direction | Gene")

    cbar_label = "row z-score" if row_zscore else ("Expression change" if normalize_to_stage0 else "Expression")
    aux_axes = [a for a in fig.axes if a is not heatmap_ax and a is not groupby_ax]
    if aux_axes:
        aux_axes[0].set_ylabel(cbar_label)

    fig.set_dpi(dpi)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, heatmap_ax, matrix


def plot_dynamic_marker_heatmap_track_panels(
    adata,
    *,
    tracks: Optional[Sequence[str]] = None,
    directions: Sequence[str] = ("increasing", "decreasing"),
    top_n: int = 5,
    per_track_total: bool = True,
    protein_coding_only: bool = False,
    require_genes_in_var: bool = False,
    noncoding_prefixes: Sequence[str] = ("AC", "AL", "AP", "LINC"),
    exclude_dot_genes: bool = False,
    row_zscore: bool = True,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    cell_groupby: str = "stage",
    restrict_to_track_cells: bool = True,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = -1.0,
    vmax: Optional[float] = 1.0,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.2, 4.2),
    dpi: int = 100,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot one dynamic-marker heatmap panel per track (Fig. 4A-style multi-panel layout).

    Each panel is rendered with ``sc.pl.heatmap`` using cells restricted to that track.
    """
    progression = _get_progression_markers(adata)
    chosen_tracks = list(tracks) if tracks is not None else _sorted_stage_labels(list(progression.keys()))
    if not chosen_tracks:
        raise ValueError("No tracks available for plotting.")

    direction_list = [str(d) for d in directions if str(d)]
    if not direction_list:
        raise ValueError("directions must be non-empty.")
    groupby_mode = str(cell_groupby).strip().lower()
    if groupby_mode not in {"stage", "stage_cluster", "cluster"}:
        raise ValueError("cell_groupby must be one of: stage, stage_cluster, cluster")

    prefixes = tuple(str(p).upper() for p in noncoding_prefixes if str(p))
    var_gene_set = set(str(g) for g in adata.var_names) if require_genes_in_var else None
    sc = _prepare_scanpy()

    panel_imgs: list[np.ndarray] = []
    panel_titles: list[str] = []
    meta_records: list[dict[str, Any]] = []

    for track in chosen_tracks:
        # Select dynamic markers for this track.
        selected_tables: list[pd.DataFrame] = []
        if per_track_total:
            n_dirs = len(direction_list)
            if top_n is not None and top_n > 0:
                base = int(top_n) // n_dirs
                rem = int(top_n) % n_dirs
                quotas = {d: base + (1 if idx < rem else 0) for idx, d in enumerate(direction_list)}
            else:
                quotas = {d: top_n for d in direction_list}
            for direction in direction_list:
                quota = quotas.get(direction, 0)
                if quota is not None and quota <= 0:
                    continue
                table = extract_dynamic_marker_table(
                    adata,
                    track=str(track),
                    direction=str(direction),
                    top_n=quota,
                    sort_by="rank",
                )
                if table.empty:
                    continue
                selected_tables.append(table)
        else:
            for direction in direction_list:
                table = extract_dynamic_marker_table(
                    adata,
                    track=str(track),
                    direction=str(direction),
                    top_n=top_n,
                    sort_by="rank",
                )
                if table.empty:
                    continue
                selected_tables.append(table)

        if not selected_tables:
            continue

        marker_genes: list[str] = []
        seen_genes: set[str] = set()
        stage_cols: list[str] = []
        for table in selected_tables:
            if not stage_cols:
                stage_cols = _progression_stage_columns(table)
            for _, row in table.iterrows():
                gene = str(row.get("gene", "gene"))
                gene_upper = gene.upper()
                if var_gene_set is not None and gene not in var_gene_set:
                    continue
                if protein_coding_only:
                    if prefixes and any(gene_upper.startswith(pref) for pref in prefixes):
                        continue
                    if exclude_dot_genes and "." in gene:
                        continue
                if gene not in seen_genes:
                    seen_genes.add(gene)
                    marker_genes.append(gene)

        if not marker_genes:
            continue

        # Restrict to this track's cells.
        obs_mask = np.ones(adata.n_obs, dtype=bool)
        if stage_key in adata.obs.columns and stage_cols:
            valid_stages = set(str(col).replace("stage", "") for col in stage_cols)
            obs_mask &= adata.obs[stage_key].astype(str).isin(valid_stages).to_numpy(dtype=bool)
        if restrict_to_track_cells and stage_key in adata.obs.columns and cluster_key in adata.obs.columns:
            allowed_pairs: set[tuple[str, str]] = set()
            for s_idx, token in enumerate(str(track).split("-")):
                for cluster_token in str(token).split("n"):
                    cluster_id = str(cluster_token).strip()
                    if cluster_id:
                        allowed_pairs.add((str(s_idx), cluster_id))
            if allowed_pairs:
                stages = adata.obs[stage_key].astype(str).to_numpy()
                clusters = adata.obs[cluster_key].astype(str).to_numpy()
                pair_mask = np.fromiter(
                    ((stages[i], clusters[i]) in allowed_pairs for i in range(adata.n_obs)),
                    dtype=bool,
                    count=adata.n_obs,
                )
                obs_mask &= pair_mask

        cell_adata = adata[obs_mask, marker_genes].to_memory() if adata.isbacked else adata[obs_mask, marker_genes].copy()
        if cell_adata.n_obs == 0:
            continue

        # Group labels in panel.
        has_stage = stage_key in cell_adata.obs.columns
        has_cluster = cluster_key in cell_adata.obs.columns
        if has_stage:
            stage_vals = cell_adata.obs[stage_key].astype(str)
            stage_order = [str(v) for v in _sorted_stage_labels(stage_vals.unique().tolist())]
            stage_labels = [f"stage {s}" for s in stage_order]
            stage_label_values = stage_vals.map(lambda s: f"stage {s}")
            cell_adata.obs["_stage_label"] = pd.Categorical(stage_label_values, categories=stage_labels, ordered=True)
        if has_cluster:
            cluster_vals = cell_adata.obs[cluster_key].astype(str)
            cluster_order = [str(v) for v in _sorted_stage_labels(cluster_vals.unique().tolist())]
            cell_adata.obs[cluster_key] = pd.Categorical(cluster_vals, categories=cluster_order, ordered=True)

        if groupby_mode == "stage_cluster" and has_stage and has_cluster:
            stage_vals = cell_adata.obs[stage_key].astype(str)
            cluster_vals = cell_adata.obs[cluster_key].astype(str)
            group_vals = stage_vals + "|" + cluster_vals
            order_df = pd.DataFrame({"stage": stage_vals, "cluster": cluster_vals, "group": group_vals}).drop_duplicates()
            order_df["stage_sort"] = order_df["stage"].apply(_safe_numeric_sort_key)
            order_df["cluster_sort"] = order_df["cluster"].apply(_safe_numeric_sort_key)
            order_df = order_df.sort_values(["stage_sort", "cluster_sort"])
            group_order = order_df["group"].tolist()
            cell_adata.obs["_dynamic_group"] = pd.Categorical(group_vals, categories=group_order, ordered=True)
            groupby_key: str = "_dynamic_group"
        elif groupby_mode == "cluster" and has_cluster:
            groupby_key = cluster_key
        elif has_stage:
            groupby_key = "_stage_label"
        elif has_cluster:
            groupby_key = cluster_key
        else:
            cell_adata.obs["_dynamic_group"] = pd.Categorical(
                [str(i) for i in range(cell_adata.n_obs)],
                categories=[str(i) for i in range(cell_adata.n_obs)],
                ordered=True,
            )
            groupby_key = "_dynamic_group"

        # Z-score across all cells in adata for the selected genes.
        if row_zscore:
            full_matrix = adata[:, marker_genes].X
            full_arr = full_matrix.toarray() if hasattr(full_matrix, "toarray") else np.asarray(full_matrix)
            if full_arr.ndim != 2:
                full_arr = np.asarray(full_arr).reshape(adata.n_obs, len(marker_genes))
            gene_means = full_arr.mean(axis=0, keepdims=True)
            gene_stds = full_arr.std(axis=0, keepdims=True)
            gene_stds[gene_stds == 0.0] = 1.0
            subset_matrix = cell_adata.X
            subset_arr = subset_matrix.toarray() if hasattr(subset_matrix, "toarray") else np.asarray(subset_matrix)
            if subset_arr.ndim != 2:
                subset_arr = np.asarray(subset_arr).reshape(cell_adata.n_obs, len(marker_genes))
            cell_adata.layers["scaled"] = (subset_arr - gene_means) / gene_stds
            layer_name: Optional[str] = "scaled"
        else:
            layer_name = None

        hm_axes = sc.pl.heatmap(
            cell_adata,
            var_names=marker_genes,
            groupby=groupby_key,
            layer=layer_name,
            cmap=cmap,
            dendrogram=False,
            swap_axes=True,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize_per_panel,
            show=False,
        )
        heatmap_ax = hm_axes.get("heatmap_ax") if isinstance(hm_axes, dict) else hm_axes
        groupby_ax = hm_axes.get("groupby_ax") if isinstance(hm_axes, dict) else None
        panel_fig = heatmap_ax.figure if heatmap_ax is not None else (groupby_ax.figure if groupby_ax is not None else plt.gcf())
        if heatmap_ax is not None:
            heatmap_ax.set_title(f"Track {track}", fontsize=10)
        if groupby_ax is not None and groupby_key == "_stage_label":
            groupby_ax.set_ylabel("stage")

        panel_fig.canvas.draw()
        img = np.asarray(panel_fig.canvas.buffer_rgba())[:, :, :3].copy()
        panel_imgs.append(img)
        panel_titles.append(f"Track {track}")
        meta_records.append(
            {
                "track": str(track),
                "n_cells": int(cell_adata.n_obs),
                "n_markers": int(len(marker_genes)),
            }
        )
        plt.close(panel_fig)

    if not panel_imgs:
        raise ValueError("No track panels were generated; check filters and available tracks.")

    n_panels = len(panel_imgs)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=dpi,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    for i, ax in enumerate(flat_axes):
        if i < n_panels:
            ax.imshow(panel_imgs[i])
            ax.axis("off")
            ax.set_title(panel_titles[i], fontsize=10)
        else:
            ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, pd.DataFrame.from_records(meta_records)


def _get_hierarchical_marker_entry(adata, stage: Any, cluster: Any, level: Any) -> Mapping[str, Any]:
    if "hcmarkers" not in adata.uns:
        raise KeyError("Missing adata.uns['hcmarkers']. Run hierarchical marker discovery first.")
    hc = adata.uns["hcmarkers"]
    stage_key = _resolve_key_by_string(hc, stage, "stage")
    stage_bucket = hc[stage_key]
    if "markers" not in stage_bucket:
        raise KeyError(f"Missing markers for stage {stage_key!r}.")
    marker_bucket = stage_bucket["markers"]
    cluster_key = _resolve_key_by_string(marker_bucket, cluster, "cluster")
    cluster_bucket = marker_bucket[cluster_key]
    level_key = _resolve_key_by_string(cluster_bucket, level, "level")
    return cluster_bucket[level_key]


def extract_hierarchical_marker_table(
    adata,
    *,
    stage: Any,
    cluster: Any,
    level: Any,
    direction: str = "up",
    top_n: Optional[int] = None,
    sort_by: str = "log_fold_changes",
) -> pd.DataFrame:
    """
    Extract one hierarchical-marker table for a given stage/cluster/level branch.
    """
    entry = _get_hierarchical_marker_entry(adata, stage=stage, cluster=cluster, level=level)
    chosen = entry.get("chosen", {})
    direction_s = str(direction).lower()
    if direction_s in {"up", "pos", "positive", "increasing"}:
        preferred = "pos"
        sign = 1
    elif direction_s in {"down", "neg", "negative", "decreasing"}:
        preferred = "neg"
        sign = -1
    else:
        raise ValueError("direction must be one of: up/down/positive/negative/increasing/decreasing")

    raw = chosen.get(preferred)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        alt = chosen.get("pos") if preferred == "neg" else chosen.get("neg")
        if isinstance(alt, pd.DataFrame) and not alt.empty:
            raw = alt.copy()
        else:
            return pd.DataFrame()

    table = raw.copy()
    table["gene"] = table.index.astype(str)
    if "log_fold_changes" in table.columns:
        table["log_fold_changes"] = pd.to_numeric(table["log_fold_changes"], errors="coerce")
        if sign > 0:
            table = table[table["log_fold_changes"] > 0].copy()
        else:
            table = table[table["log_fold_changes"] < 0].copy()
    if "pvals_adj" in table.columns:
        table["pvals_adj"] = pd.to_numeric(table["pvals_adj"], errors="coerce")

    if sort_by in table.columns:
        ascending = sort_by != "log_fold_changes"
        if sort_by == "log_fold_changes":
            table = table.reindex(table["log_fold_changes"].abs().sort_values(ascending=False).index)
        else:
            table = table.sort_values(sort_by, ascending=ascending, na_position="last")
    if top_n is not None and top_n > 0:
        table = table.head(int(top_n)).copy()
    table = table.reset_index(drop=True)
    return table


def plot_hierarchical_marker_branch_heatmap(
    adata,
    *,
    stage: Any,
    cluster: Any,
    level: Any,
    direction: str = "up",
    top_n: int = 20,
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    celltype_key: Optional[str] = "ident",
    layer: Optional[str] = None,
    zscore: bool = True,
    highlight_selected_clusters: bool = True,
    selected_label_prefix: str = "★ ",
    selected_cluster_color: str = "#E64B35",
    selected_cluster_fontweight: str = "bold",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (11.0, 6.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Heatmap for hierarchical branch markers comparing selected cluster(s) vs sibling scope (Fig. 4c-e style).
    """
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if cluster_key not in adata.obs:
        raise KeyError(f"Missing cluster key in adata.obs: {cluster_key}")
    if celltype_key and celltype_key not in adata.obs:
        raise KeyError(f"Missing celltype key in adata.obs: {celltype_key}")

    markers = extract_hierarchical_marker_table(
        adata,
        stage=stage,
        cluster=cluster,
        level=level,
        direction=direction,
        top_n=top_n,
        sort_by="log_fold_changes",
    )
    if markers.empty:
        raise ValueError("No hierarchical markers available after filtering.")

    entry = _get_hierarchical_marker_entry(adata, stage=stage, cluster=cluster, level=level)
    ref = entry.get("ref", [[], []])
    selected_clusters = [str(v) for v in (ref[0] if len(ref) > 0 else [])]
    sibling_clusters = [str(v) for v in (ref[1] if len(ref) > 1 else []) if str(v) not in selected_clusters]
    ordered_clusters = selected_clusters + sorted(sibling_clusters, key=_safe_numeric_sort_key)
    if not ordered_clusters:
        ordered_clusters = sorted(adata.obs[cluster_key].astype(str).unique().tolist(), key=_safe_numeric_sort_key)

    stage_s = str(stage)
    stage_mask = adata.obs[stage_key].astype(str) == stage_s
    stage_adata = adata[stage_mask].to_memory() if adata.isbacked else adata[stage_mask].copy()
    stage_adata.obs[cluster_key] = stage_adata.obs[cluster_key].astype(str)

    genes = [g for g in markers["gene"].astype(str).tolist() if g in stage_adata.var_names]
    if not genes:
        raise ValueError("None of the selected markers are present in adata.var_names.")

    matrix_rows: list[np.ndarray] = []
    col_labels: list[str] = []
    col_is_selected: list[bool] = []
    valid_clusters: list[str] = []
    selected_set = {str(v) for v in selected_clusters}
    for clu in ordered_clusters:
        sub = stage_adata[stage_adata.obs[cluster_key] == str(clu), genes]
        if sub.n_obs == 0:
            continue
        data = sub.layers[layer] if layer else sub.X
        if hasattr(data, "toarray"):
            data = data.toarray()
        mean_profile = np.asarray(data).mean(axis=0).reshape(-1)
        matrix_rows.append(mean_profile)
        valid_clusters.append(str(clu))
        is_selected = str(clu) in selected_set
        if celltype_key:
            dominant = _dominant_label(sub.obs[celltype_key], fallback="NA")
            label = f"{clu} | {dominant}"
        else:
            label = str(clu)
        if highlight_selected_clusters and is_selected and selected_label_prefix:
            label = f"{selected_label_prefix}{label}"
        col_labels.append(label)
        col_is_selected.append(is_selected)

    if not matrix_rows:
        raise ValueError("No stage/cluster cells available for the selected branch.")

    matrix = pd.DataFrame(np.vstack(matrix_rows).T, index=genes, columns=col_labels)
    if zscore:
        gene_std = matrix.std(axis=1).replace(0, 1.0)
        matrix = matrix.sub(matrix.mean(axis=1), axis=0).div(gene_std, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    sns.heatmap(matrix, cmap=cmap, center=0 if zscore else None, ax=ax, cbar_kws={"label": "z-score" if zscore else "mean expression"})
    xlabel = "Cluster" if not celltype_key else "Cluster | Dominant cell type"
    if highlight_selected_clusters and any(col_is_selected):
        xlabel += f" ({selected_label_prefix.strip()} selected)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marker gene")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    if highlight_selected_clusters:
        for tick, is_selected in zip(ax.get_xticklabels(), col_is_selected):
            if is_selected:
                tick.set_color(selected_cluster_color)
                tick.set_fontweight(selected_cluster_fontweight)
    ax.set_title(f"Hierarchical markers: stage={stage_s}, cluster={cluster}, level={level}, direction={direction}")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, matrix, markers


def _extract_temporal_edges(adata, edges_key: str = "edges") -> pd.DataFrame:
    if edges_key not in adata.uns:
        raise KeyError(f"Missing adata.uns['{edges_key}'].")
    bucket = adata.uns[edges_key]
    if not isinstance(bucket, Mapping):
        raise TypeError(f"adata.uns['{edges_key}'] must be a mapping.")

    rows: list[dict[str, Any]] = []
    for key, pairs in bucket.items():
        try:
            stage_i = int(str(key))
        except (TypeError, ValueError):
            continue
        if not isinstance(pairs, Sequence):
            continue
        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) < 2:
                continue
            rows.append(
                {
                    "src_stage_idx": int(stage_i),
                    "src_cluster": str(pair[0]),
                    "dst_stage_idx": int(stage_i) + 1,
                    "dst_cluster": str(pair[1]),
                }
            )
    edges = pd.DataFrame.from_records(rows).drop_duplicates()
    if edges.empty:
        raise ValueError(f"No valid edges found in adata.uns['{edges_key}'].")
    return edges


def _build_track_paths_from_edges(edges: pd.DataFrame, stage_count: int) -> list[list[tuple[int, str]]]:
    """
    Build full stage-to-stage paths from temporal edges as (stage_idx, cluster) tuples.
    """
    if stage_count < 2:
        return []
    required = {"src_stage_idx", "dst_stage_idx", "src_cluster", "dst_cluster"}
    if not required.issubset(edges.columns):
        raise KeyError(f"edges DataFrame must include columns: {sorted(required)}")

    adjacency: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for _, row in edges.iterrows():
        src = (int(row["src_stage_idx"]), str(row["src_cluster"]))
        dst = (int(row["dst_stage_idx"]), str(row["dst_cluster"]))
        adjacency.setdefault(src, []).append(dst)
    for key in list(adjacency.keys()):
        uniq = sorted({tuple(v) for v in adjacency[key]}, key=lambda t: (t[0], _safe_numeric_sort_key(t[1])))
        adjacency[key] = [(int(s), str(c)) for s, c in uniq]

    starts = sorted(
        edges.loc[edges["src_stage_idx"] == 0, "src_cluster"].astype(str).unique().tolist(),
        key=_safe_numeric_sort_key,
    )

    raw_paths: list[list[tuple[int, str]]] = []

    def _dfs(node: tuple[int, str], path: list[tuple[int, str]]) -> None:
        stage_idx = int(node[0])
        if stage_idx == stage_count - 1:
            raw_paths.append(path.copy())
            return
        children = [c for c in adjacency.get(node, []) if int(c[0]) == stage_idx + 1]
        if not children:
            return
        for child in children:
            _dfs(child, path + [child])

    for clu in starts:
        _dfs((0, str(clu)), [(0, str(clu))])

    # Keep unique full-length cluster sequences only.
    dedup: dict[tuple[str, ...], list[tuple[int, str]]] = {}
    for path in raw_paths:
        if len(path) != stage_count:
            continue
        seq = tuple(str(c) for _, c in path)
        if seq not in dedup:
            dedup[seq] = path
    out = list(dedup.values())
    out.sort(key=lambda p: tuple(_safe_numeric_sort_key(c) for _, c in p))
    return out


def _build_track_specs_from_progression_markers(
    adata,
    *,
    stage_order: Sequence[str],
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    edges: Optional[pd.DataFrame] = None,
) -> list[dict[str, Any]]:
    """
    Build track specs from adata.uns['progressionMarkers'] to preserve canonical UNAGI tracks,
    including intra-track branching (tokens like "10n16").
    """
    pm = adata.uns.get("progressionMarkers")
    if not isinstance(pm, Mapping) or not pm:
        return []

    specs: list[dict[str, Any]] = []
    edge_lookup: dict[int, set[tuple[str, str]]] = {}
    if edges is not None and not edges.empty:
        for _, row in edges.iterrows():
            s_idx = int(row["src_stage_idx"])
            edge_lookup.setdefault(s_idx, set()).add((str(row["src_cluster"]), str(row["dst_cluster"])))

    for track_key in sorted([str(k) for k in pm.keys()], key=_safe_numeric_sort_key):
        tokens = str(track_key).split("-")
        if len(tokens) != len(stage_order):
            continue
        stage_options: list[list[str]] = []
        for s_idx, token in enumerate(tokens):
            options = sorted({str(t) for t in str(token).split("n") if str(t) != ""}, key=_safe_numeric_sort_key)
            if not options:
                options = [str(token)]
            stage_options.append(options)

        edge_pairs: list[tuple[int, str, str]] = []
        for s_idx in range(len(stage_options) - 1):
            left = stage_options[s_idx]
            right = stage_options[s_idx + 1]
            selected: list[tuple[str, str]] = []
            if s_idx in edge_lookup:
                selected = sorted(
                    [
                        (src, dst)
                        for src, dst in edge_lookup[s_idx]
                        if src in left and dst in right
                    ],
                    key=lambda p: (_safe_numeric_sort_key(p[0]), _safe_numeric_sort_key(p[1])),
                )
            # Fallback: if edges are unavailable for this pair, preserve branch possibilities.
            if not selected:
                selected = [
                    (src, dst)
                    for src in left
                    for dst in right
                ]
            for src, dst in selected:
                edge_pairs.append((int(s_idx), str(src), str(dst)))

        specs.append(
            {
                "track_key": str(track_key),
                "stage_options": stage_options,
                "edge_pairs": edge_pairs,
            }
        )
    return specs


def plot_temporal_dynamics_graph(
    adata,
    *,
    edges_key: str = "edges",
    stage_key: str = "stage",
    cluster_key: str = "leiden",
    celltype_key: str = "ident",
    layout_mode: str = "track_separated",
    track_source: str = "auto",
    stage_label_map: Optional[Mapping[Any, str]] = None,
    stage_label_prefix: str = "Stage ",
    node_palette: Optional[Mapping[str, str]] = None,
    orientation: str = "vertical",
    track_gap: float = 1.8,
    branch_node_gap: float = 0.36,
    show_track_labels: bool = True,
    track_label_prefix: str = "Track ",
    track_label_style: str = "paper_like",
    track_label_map: Optional[Mapping[str, str]] = None,
    track_label_rotation: float = 50.0,
    track_label_fontsize: float = 7.5,
    track_label_offset: float = 0.72,
    max_tracks: Optional[int] = 12,
    node_size_mode: str = "constant",
    node_size: float = 280.0,
    min_node_size: float = 180.0,
    max_node_size: float = 1100.0,
    stage_gap: float = 1.4,
    cluster_gap: float = 1.25,
    node_alpha: float = 0.95,
    edge_alpha: float = 0.45,
    edge_width: float = 1.4,
    show_node_labels: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    min_figsize: Tuple[float, float] = (6.0, 6.0),
    max_figsize: Tuple[float, float] = (18.0, 24.0),
    stage_size_scale: float = 1.05,
    node_size_scale: float = 0.72,
    legend_width_padding: float = 1.8,
    dpi: int = 100,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot stage-resolved temporal dynamics graph from UNAGI edges (Fig. 3a-style).

    Defaults emphasize manuscript-style readability:
    - vertical stage progression
    - track-separated lanes with straighter trajectories (prefers progressionMarkers)
    - constant node sizes
    - larger intra-stage spacing to reduce overlap
    - auto figure-size scaling from stage count and nodes-per-stage
    """
    for key_name, column in [(stage_key, stage_key), (cluster_key, cluster_key), (celltype_key, celltype_key)]:
        if column not in adata.obs:
            raise KeyError(f"Missing {key_name!r} in adata.obs")

    obs = adata.obs[[stage_key, cluster_key, celltype_key]].copy()
    obs[stage_key] = obs[stage_key].astype(str)
    obs[cluster_key] = obs[cluster_key].astype(str)
    obs[celltype_key] = obs[celltype_key].astype(str)

    stage_order = _sorted_stage_labels(obs[stage_key].unique().tolist())
    stage_order = [str(v) for v in stage_order]
    if len(stage_order) < 2:
        raise ValueError("Need at least two stages to plot temporal dynamics.")
    idx_to_stage = {i: stage for i, stage in enumerate(stage_order)}

    edges = _extract_temporal_edges(adata, edges_key=edges_key)
    edges = edges[
        edges["src_stage_idx"].isin(idx_to_stage.keys())
        & edges["dst_stage_idx"].isin(idx_to_stage.keys())
    ].copy()
    if edges.empty:
        raise ValueError("No temporal edges overlap with available stage labels.")
    edges["src_stage"] = edges["src_stage_idx"].map(idx_to_stage)
    edges["dst_stage"] = edges["dst_stage_idx"].map(idx_to_stage)
    edges["src_node"] = list(zip(edges["src_stage"], edges["src_cluster"]))
    edges["dst_node"] = list(zip(edges["dst_stage"], edges["dst_cluster"]))

    node_counts = (
        obs.groupby([stage_key, cluster_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .rename(columns={stage_key: "stage", cluster_key: "cluster"})
    )
    node_ct = (
        obs.groupby([stage_key, cluster_key], observed=False)[celltype_key]
        .agg(lambda s: _dominant_label(pd.Series(s)))
        .rename("cell_type")
        .reset_index()
        .rename(columns={stage_key: "stage", cluster_key: "cluster"})
    )
    nodes = node_counts.merge(node_ct, on=["stage", "cluster"], how="outer")
    nodes["n_cells"] = pd.to_numeric(nodes["n_cells"], errors="coerce").fillna(0.0)

    edge_nodes = pd.DataFrame(
        {
            "stage": pd.concat([edges["src_stage"], edges["dst_stage"]], axis=0, ignore_index=True),
            "cluster": pd.concat([edges["src_cluster"], edges["dst_cluster"]], axis=0, ignore_index=True),
        }
    ).drop_duplicates()
    nodes = edge_nodes.merge(nodes, on=["stage", "cluster"], how="left")
    nodes["n_cells"] = nodes["n_cells"].fillna(0.0)
    nodes["cell_type"] = nodes["cell_type"].fillna("NA")
    nodes["node"] = list(zip(nodes["stage"], nodes["cluster"]))

    orient = str(orientation).lower()
    if orient not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    layout = str(layout_mode).lower()
    if layout not in {"track_separated", "graph"}:
        raise ValueError("layout_mode must be 'track_separated' or 'graph'.")
    track_src = str(track_source).lower()
    if track_src not in {"auto", "progression_markers", "edges"}:
        raise ValueError("track_source must be 'auto', 'progression_markers', or 'edges'.")
    label_style = str(track_label_style).lower()
    if label_style not in {"paper_like", "path", "index"}:
        raise ValueError("track_label_style must be 'paper_like', 'path', or 'index'.")
    size_mode = str(node_size_mode).lower()
    if size_mode not in {"constant", "cells"}:
        raise ValueError("node_size_mode must be 'constant' or 'cells'.")

    stage_count = len(stage_order)
    node_counts_by_stage = nodes.groupby("stage", observed=False)["cluster"].nunique()
    max_nodes_per_stage = int(node_counts_by_stage.max()) if not node_counts_by_stage.empty else 1
    track_count = 0
    use_track_layout = layout == "track_separated"

    node_meta_lookup = (
        nodes.set_index(["stage", "cluster"])[["n_cells", "cell_type"]].to_dict(orient="index")
        if not nodes.empty
        else {}
    )

    track_specs: list[dict[str, Any]] = []
    max_branch_per_track = 1
    lane_step = float(track_gap)
    if use_track_layout:
        if track_src in {"auto", "progression_markers"}:
            track_specs = _build_track_specs_from_progression_markers(
                adata,
                stage_order=stage_order,
                stage_key=stage_key,
                cluster_key=cluster_key,
                edges=edges,
            )
        if not track_specs and track_src in {"auto", "edges"}:
            edge_paths = _build_track_paths_from_edges(edges, stage_count=stage_count)
            track_specs = []
            for p in edge_paths:
                stage_options = [[str(clu)] for _, clu in p]
                edge_pairs = []
                for i in range(len(p) - 1):
                    edge_pairs.append((int(p[i][0]), str(p[i][1]), str(p[i + 1][1])))
                track_specs.append(
                    {
                        "track_key": "-".join([str(c) for _, c in p]),
                        "stage_options": stage_options,
                        "edge_pairs": edge_pairs,
                    }
                )

        if track_specs and max_tracks is not None and int(max_tracks) > 0 and len(track_specs) > int(max_tracks):
            def _path_score(spec: dict[str, Any]) -> float:
                score = 0.0
                stage_options = spec.get("stage_options", [])
                for s_idx, options in enumerate(stage_options):
                    stage_name = idx_to_stage.get(int(s_idx), str(s_idx))
                    if not options:
                        continue
                    score += max(
                        float(node_meta_lookup.get((str(stage_name), str(clu)), {}).get("n_cells", 0.0))
                        for clu in options
                    )
                return score
            track_specs = sorted(track_specs, key=_path_score, reverse=True)[: int(max_tracks)]
        track_specs = sorted(track_specs, key=lambda d: _safe_numeric_sort_key(str(d.get("track_key", ""))))
        track_count = len(track_specs)
        if track_specs:
            max_branch_per_track = max(
                max(len(stage_opts) for stage_opts in spec.get("stage_options", [[]]))
                for spec in track_specs
            )
            max_branch_per_track = max(1, int(max_branch_per_track))
            lane_step = max(float(track_gap), 0.8 + float(branch_node_gap) * float(max(0, max_branch_per_track - 1)))
        if track_count == 0:
            use_track_layout = False

    if figsize is None:
        safe_stage_gap = max(0.5, float(stage_gap))
        safe_cluster_gap = max(0.5, float(lane_step) if use_track_layout else float(cluster_gap))
        stage_span = max(2, stage_count)
        node_span = max(2, track_count if use_track_layout else max_nodes_per_stage)
        if orient == "vertical":
            width = 2.4 + node_span * float(node_size_scale) * safe_cluster_gap + float(legend_width_padding)
            height = 2.0 + stage_span * float(stage_size_scale) * safe_stage_gap
        else:
            width = 2.0 + stage_span * float(stage_size_scale) * safe_stage_gap + float(legend_width_padding)
            height = 2.4 + node_span * float(node_size_scale) * safe_cluster_gap
        if use_track_layout and show_track_labels:
            if orient == "vertical":
                height += 0.9
            else:
                width += 1.0
        width = float(np.clip(width, min_figsize[0], max_figsize[0]))
        height = float(np.clip(height, min_figsize[1], max_figsize[1]))
        figsize_use = (width, height)
    else:
        figsize_use = (float(figsize[0]), float(figsize[1]))

    positions: Dict[Any, Tuple[float, float]] = {}
    track_labels: list[str] = []
    if use_track_layout:
        track_nodes_rows: list[dict[str, Any]] = []
        track_edge_rows: list[dict[str, Any]] = []
        lane_half = 0.5 * max(0, track_count - 1)
        for t_idx, spec in enumerate(track_specs):
            raw_track_key = str(spec.get("track_key", f"{t_idx+1}"))
            stage_options = spec.get("stage_options", [])
            edge_pairs = spec.get("edge_pairs", [])
            lane_pos = (float(t_idx) - lane_half) * float(lane_step)
            if track_label_map and raw_track_key in track_label_map:
                track_label = str(track_label_map[raw_track_key])
            elif label_style == "path":
                track_label = f"{track_label_prefix}{t_idx + 1}: {raw_track_key}"
            elif label_style == "index":
                track_label = f"{track_label_prefix}{t_idx + 1}"
            else:
                # Paper-like naming: <cell-type>-<first stage-0 cluster id>.
                root_token = str(raw_track_key).split("-")[0] if raw_track_key else ""
                root_cluster = str(root_token).split("n")[0] if root_token else ""
                root_options = stage_options[0] if stage_options else []
                if not root_cluster and root_options:
                    root_cluster = sorted({str(c) for c in root_options}, key=_safe_numeric_sort_key)[0]
                root_stage = idx_to_stage.get(0, str(stage_order[0]))
                root_ident = str(node_meta_lookup.get((str(root_stage), str(root_cluster)), {}).get("cell_type", "Track"))
                base = _abbreviate_track_label(root_ident)
                track_label = f"{base}-{root_cluster}" if root_cluster else f"{base}-{t_idx + 1}"
            track_labels.append(track_label)
            node_key_lookup: dict[tuple[int, str], tuple[int, int, str]] = {}

            for s_idx in range(stage_count):
                options = stage_options[s_idx] if s_idx < len(stage_options) else []
                options = sorted({str(c) for c in options}, key=_safe_numeric_sort_key)
                if not options:
                    continue
                if len(options) == 1:
                    offsets = np.array([0.0], dtype=float)
                else:
                    half = 0.5 * (len(options) - 1)
                    offsets = np.linspace(-half, half, len(options), dtype=float) * float(branch_node_gap)
                for clu, off in zip(options, offsets):
                    point_key = (int(t_idx), int(s_idx), str(clu))
                    stage_name = idx_to_stage.get(int(s_idx), str(s_idx))
                    if orient == "vertical":
                        positions[point_key] = (float(lane_pos) + float(off), -float(s_idx) * float(stage_gap))
                    else:
                        positions[point_key] = (-float(s_idx) * float(stage_gap), float(lane_pos) + float(off))
                    node_key_lookup[(int(s_idx), str(clu))] = point_key
                    meta = node_meta_lookup.get((str(stage_name), str(clu)), {})
                    track_nodes_rows.append(
                        {
                            "node": point_key,
                            "track_idx": int(t_idx),
                            "track_key": raw_track_key,
                            "track_label": track_label,
                            "stage_idx": int(s_idx),
                            "stage": str(stage_name),
                            "cluster": str(clu),
                            "n_cells": float(meta.get("n_cells", 0.0)),
                            "cell_type": str(meta.get("cell_type", "NA")),
                        }
                    )

            for s_idx, src_clu, dst_clu in edge_pairs:
                src_key = node_key_lookup.get((int(s_idx), str(src_clu)))
                dst_key = node_key_lookup.get((int(s_idx) + 1, str(dst_clu)))
                if src_key is None or dst_key is None:
                    continue
                track_edge_rows.append(
                    {
                        "track_idx": int(t_idx),
                        "track_key": raw_track_key,
                        "track_label": track_label,
                        "src_node": src_key,
                        "dst_node": dst_key,
                        "src_stage_idx": int(s_idx),
                        "dst_stage_idx": int(s_idx) + 1,
                        "src_stage": str(idx_to_stage.get(int(s_idx), str(s_idx))),
                        "dst_stage": str(idx_to_stage.get(int(s_idx) + 1, str(int(s_idx) + 1))),
                        "src_cluster": str(src_clu),
                        "dst_cluster": str(dst_clu),
                    }
                )
        nodes = pd.DataFrame.from_records(track_nodes_rows).drop_duplicates(subset=["node"])
        edges = pd.DataFrame.from_records(track_edge_rows)
    else:
        for s_idx, stage in enumerate(stage_order):
            stage_nodes = sorted(nodes.loc[nodes["stage"] == stage, "cluster"].astype(str).unique().tolist(), key=_safe_numeric_sort_key)
            if not stage_nodes:
                continue
            if s_idx > 0:
                incoming = edges[edges["dst_stage"] == stage]
                scores: dict[str, float] = {}
                for clu in stage_nodes:
                    parents = incoming.loc[incoming["dst_cluster"] == clu, ["src_stage", "src_cluster"]].itertuples(index=False)
                    parent_axis_pos = [positions[(str(p_stage), str(p_clu))][0] for p_stage, p_clu in parents if (str(p_stage), str(p_clu)) in positions]
                    if parent_axis_pos:
                        scores[clu] = float(np.mean(parent_axis_pos))
                    else:
                        scores[clu] = float("inf")
                stage_nodes = sorted(stage_nodes, key=lambda c: (scores.get(c, float("inf")), _safe_numeric_sort_key(c)))

            if len(stage_nodes) == 1:
                axis_vals = np.array([0.0], dtype=float)
            else:
                half = 0.5 * (len(stage_nodes) - 1)
                axis_vals = np.linspace(-half, half, len(stage_nodes), dtype=float) * float(cluster_gap)

            # stage0 at top, later stages lower on the y-axis in vertical mode
            stage_axis = -float(s_idx) * float(stage_gap)
            for clu, axis_pos in zip(stage_nodes, axis_vals):
                if orient == "vertical":
                    positions[(stage, clu)] = (float(axis_pos), stage_axis)
                else:
                    positions[(stage, clu)] = (stage_axis, float(axis_pos))

    nodes["x"] = [positions[k][0] for k in nodes["node"]]
    nodes["y"] = [positions[k][1] for k in nodes["node"]]

    count_vals = nodes["n_cells"].to_numpy(dtype=float).reshape(-1)
    if size_mode == "constant":
        node_sizes = np.full_like(count_vals, float(node_size), dtype=float)
    else:
        if np.allclose(np.max(count_vals), np.min(count_vals)):
            node_sizes = np.full_like(count_vals, 0.5 * (min_node_size + max_node_size), dtype=float)
        else:
            node_sizes = np.interp(
                count_vals,
                (float(np.min(count_vals)), float(np.max(count_vals))),
                (float(min_node_size), float(max_node_size)),
            )
    nodes["node_size"] = node_sizes

    cell_type_categories = sorted(obs[celltype_key].astype(str).unique().tolist(), key=_safe_numeric_sort_key)
    palette = build_palette(cell_type_categories, preset=node_palette)
    nodes["node_color"] = nodes["cell_type"].astype(str).map(palette)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_use, dpi=dpi)
    else:
        fig = ax.figure

    for _, row in edges.iterrows():
        src = row["src_node"] if use_track_layout else (str(row["src_stage"]), str(row["src_cluster"]))
        dst = row["dst_node"] if use_track_layout else (str(row["dst_stage"]), str(row["dst_cluster"]))
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.plot([x1, x2], [y1, y2], color="#8A8A8A", linewidth=edge_width, alpha=edge_alpha, zorder=1)

    ax.scatter(
        nodes["x"].to_numpy(dtype=float),
        nodes["y"].to_numpy(dtype=float),
        s=nodes["node_size"].to_numpy(dtype=float),
        c=nodes["node_color"].tolist(),
        alpha=node_alpha,
        linewidths=0.6,
        edgecolors="black",
        zorder=2,
    )

    if show_node_labels:
        for _, row in nodes.iterrows():
            ax.text(float(row["x"]), float(row["y"]), str(row["cluster"]), ha="center", va="center", fontsize=7, color="black", zorder=3)

    stage_tick_labels = []
    stage_ticks = []
    for idx, stage in enumerate(stage_order):
        mapped = None
        if stage_label_map:
            if stage in stage_label_map:
                mapped = stage_label_map[stage]
            elif idx in stage_label_map:
                mapped = stage_label_map[idx]
        if mapped is not None:
            stage_label = str(mapped)
        else:
            stage_s = str(stage)
            if stage_s.lower().startswith("stage"):
                stage_label = stage_s
            else:
                stage_label = f"{stage_label_prefix}{stage_s}"
        stage_tick_labels.append(stage_label)
        stage_ticks.append(-float(idx) * float(stage_gap))

    if orient == "vertical":
        ax.set_yticks(stage_ticks)
        ax.set_yticklabels(stage_tick_labels)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("Stage")
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.grid(axis="y", alpha=0.18)
    else:
        ax.set_xticks(stage_ticks)
        ax.set_xticklabels(stage_tick_labels)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Track lane")
        ax.grid(axis="x", alpha=0.18)

    # Keep nodes from touching frame edges when stage/node counts are large.
    x_vals = nodes["x"].to_numpy(dtype=float)
    y_vals = nodes["y"].to_numpy(dtype=float)
    if x_vals.size > 0 and y_vals.size > 0:
        x_pad = max(0.9, 0.5 * float(lane_step if use_track_layout else cluster_gap))
        y_pad = max(0.9, 0.5 * float(stage_gap))
        if use_track_layout and show_track_labels:
            if orient == "vertical":
                y_pad += float(track_label_offset) * float(stage_gap)
            else:
                x_pad += float(track_label_offset) * float(stage_gap)
        ax.set_xlim(float(np.min(x_vals)) - x_pad, float(np.max(x_vals)) + x_pad)
        ax.set_ylim(float(np.min(y_vals)) - y_pad, float(np.max(y_vals)) + y_pad)

    if use_track_layout and show_track_labels and track_labels:
        if orient == "vertical":
            top_y = max(stage_ticks) + float(track_label_offset) * float(stage_gap)
            for t_idx, label in enumerate(track_labels):
                lane_half = 0.5 * max(0, track_count - 1)
                lane_pos = (float(t_idx) - lane_half) * float(lane_step)
                ax.text(
                    lane_pos,
                    top_y,
                    label,
                    rotation=float(track_label_rotation),
                    ha="left",
                    va="bottom",
                    fontsize=float(track_label_fontsize),
                    color="#2F2F2F",
                    zorder=4,
                )
        else:
            right_x = max(stage_ticks) + float(track_label_offset) * float(stage_gap)
            for t_idx, label in enumerate(track_labels):
                lane_half = 0.5 * max(0, track_count - 1)
                lane_pos = (float(t_idx) - lane_half) * float(lane_step)
                ax.text(
                    right_x,
                    lane_pos,
                    label,
                    rotation=0.0,
                    ha="left",
                    va="center",
                    fontsize=float(track_label_fontsize),
                    color="#2F2F2F",
                    zorder=4,
                )

    # Remove panel boundary box for manuscript-like appearance.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Temporal dynamics graph")

    legend_items = sorted(palette.items(), key=lambda kv: kv[0])
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=color, markeredgecolor="black", label=label)
        for label, color in legend_items
    ]
    if handles:
        ax.legend(handles=handles, title=celltype_key, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, nodes, edges


def _resolve_perturb_change_key(payload: Mapping[str, Any], perturb_change: float | str) -> str:
    raw = str(perturb_change)
    if raw in payload:
        return raw
    try:
        target = float(perturb_change)
    except (TypeError, ValueError):
        raise KeyError(f"Unable to resolve perturbation key: {perturb_change}")
    for key in payload.keys():
        try:
            if abs(float(key) - target) < 1e-12:
                return str(key)
        except (TypeError, ValueError):
            continue
    raise KeyError(f"perturb_change={perturb_change} not found. Available: {list(payload.keys())}")


def extract_perturbation_table(
    adata,
    *,
    mode: str = "drug",
    perturb_change: float | str = 0.5,
    track: Optional[str] = "overall",
    prefer_top: bool = True,
) -> pd.DataFrame:
    """
    Extract perturbation ranking table from `adata.uns` for manuscript-style plotting.
    """
    mode_to_key = {
        "drug": "drug_perturbation_score",
        "pathway": "pathway_perturbation_score",
        "single_gene": "single_gene_perturbation_score",
        "gene_combo": "two_genes_perturbation_score",
    }
    if mode not in mode_to_key:
        raise ValueError(f"Unsupported mode: {mode}. Use one of {sorted(mode_to_key)}.")
    score_key = mode_to_key[mode]
    if score_key not in adata.uns:
        raise KeyError(f"Missing {score_key} in adata.uns")

    score_bucket = adata.uns[score_key]
    perturb_key = _resolve_perturb_change_key(score_bucket, perturb_change)
    track_bucket = score_bucket[perturb_key]
    selected_track = track
    if selected_track is None:
        selected_track = "overall" if "overall" in track_bucket else list(track_bucket.keys())[0]
    if selected_track not in track_bucket:
        raise KeyError(f"Track '{selected_track}' not found. Available: {list(track_bucket.keys())}")

    entry = track_bucket[selected_track]
    keys = ["top_compounds", "down_compounds"] if prefer_top else ["down_compounds", "top_compounds"]
    table_name = next((k for k in keys if k in entry), None)
    if table_name is None:
        raise KeyError(f"No perturbation table found in track '{selected_track}'. Keys: {list(entry.keys())}")

    df = pd.DataFrame.from_dict(entry[table_name]).copy()
    if df.empty:
        return df

    if "compound" in df.columns and mode == "pathway":
        df = df.rename(columns={"compound": "pathways"})
    if "compound" in df.columns and mode == "single_gene":
        df = df.rename(columns={"compound": "gene"})
    if "compound" in df.columns and mode == "gene_combo":
        df = df.rename(columns={"compound": "gene_combination"})

    if mode == "pathway" and "perturbation score" in df.columns:
        df = df.sort_values("perturbation score", ascending=False)
    elif "pval_adjusted" in df.columns:
        df = df.sort_values("pval_adjusted", ascending=True)
    elif "perturbation score" in df.columns:
        df = df.sort_values("perturbation score", ascending=False)
    df = df.reset_index(drop=True)
    return df


def plot_perturbation_ranked_bar(
    table: pd.DataFrame,
    *,
    item_col: Optional[str] = None,
    top_n: int = 20,
    score_col: str = "perturbation score",
    pval_col: str = "pval_adjusted",
    use_neglog10_pval: bool = True,
    significance: float = 0.05,
    figsize: Tuple[float, float] = (10.5, 6.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Ranked bar plot for pathway/drug perturbation hits (Fig. 5/7 style).
    """
    if table is None or table.empty:
        raise ValueError("Input table is empty.")

    candidates = ("compound", "pathways", "gene", "gene_combination", "item")
    if item_col is None:
        item_col = next((c for c in candidates if c in table.columns), None)
    if item_col is None:
        raise KeyError(f"Unable to infer item column. Expected one of {candidates}.")

    work = table.copy()
    if top_n > 0:
        work = work.iloc[:top_n].copy()

    if use_neglog10_pval and pval_col in work.columns:
        pvals = pd.to_numeric(work[pval_col], errors="coerce").fillna(1.0).clip(lower=1e-300)
        work["plot_value"] = -np.log10(pvals)
        xlabel = "-log10(adjusted p-value)"
        sig_cut = -math.log10(significance)
    else:
        if score_col not in work.columns:
            raise KeyError(f"Missing score column: {score_col}")
        work["plot_value"] = pd.to_numeric(work[score_col], errors="coerce")
        xlabel = score_col
        sig_cut = None

    work = work.sort_values("plot_value", ascending=True)
    if pval_col in work.columns:
        sig_mask = pd.to_numeric(work[pval_col], errors="coerce").fillna(1.0) <= significance
        colors = np.where(sig_mask, "#E64B35", "#91A1B0")
    else:
        colors = np.array(["#4DBBD5"] * len(work))

    wrapped_labels = [_wrap_ranked_bar_label(label) for label in work[item_col].astype(str).tolist()]
    has_wrapped_labels = any("\n" in label for label in wrapped_labels)
    line_counts = [max(1, label.count("\n") + 1) for label in wrapped_labels]
    max_line_count = max(line_counts, default=1)
    dynamic_height = max(figsize[1], 2.2 + 0.38 * max(len(work), 1))
    dynamic_width = max(figsize[0], 13.8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height), dpi=dpi)
    else:
        fig = ax.figure

    positions = np.arange(len(work))
    ax.barh(positions, work["plot_value"], color=colors, edgecolor="none")
    ax.set_yticks(positions)
    ax.set_yticklabels(wrapped_labels)
    ax.tick_params(axis="y", labelsize=8.0 if max_line_count >= 2 else 8.6, pad=4)
    ax.set_xlabel(xlabel)
    axis_label_map = {
        "pathways": "Pathways",
        "compound": "Compounds",
        "gene": "Genes",
        "gene_combination": "Gene combinations",
        "item": "Items",
    }
    ax.set_ylabel(axis_label_map.get(item_col, item_col.replace("_", " ").title()))
    ax.grid(axis="x", alpha=0.2)
    if sig_cut is not None:
        ax.axvline(sig_cut, color="#3C5488", linestyle="--", linewidth=1, alpha=0.8)
        ax.text(sig_cut, len(work) - 0.5, f"p={significance:g}", color="#3C5488", fontsize=8, ha="left", va="bottom")
    if has_wrapped_labels:
        fig.subplots_adjust(left=0.58, right=0.98, top=0.97, bottom=0.10)
    else:
        fig.tight_layout(pad=1.0)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, work


def _extract_gene_values(adata, gene: str, layer: Optional[str] = None) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene not found in adata.var_names: {gene}")
    x = adata[:, gene].layers[layer] if layer else adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x).reshape(-1)
    return arr


def plot_gene_expression_shift(
    observed_adata,
    insilico_adata,
    genes: Sequence[str],
    *,
    observed_label: str = "observed",
    insilico_label: str = "in_silico",
    layer: Optional[str] = None,
    kind: str = "box",
    figsize: Tuple[float, float] = (12.0, 4.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Compare observed vs in-silico expression for selected genes (Fig. 5/6 style).
    """
    if not genes:
        raise ValueError("genes must be a non-empty list.")
    rows = []
    for gene in genes:
        obs_vals = _extract_gene_values(observed_adata, gene, layer=layer)
        sim_vals = _extract_gene_values(insilico_adata, gene, layer=layer)
        rows.extend({"gene": gene, "condition": observed_label, "expression": float(v)} for v in obs_vals)
        rows.extend({"gene": gene, "condition": insilico_label, "expression": float(v)} for v in sim_vals)
    df = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if kind == "violin":
        sns.violinplot(data=df, x="gene", y="expression", hue="condition", split=False, ax=ax, cut=0)
    else:
        sns.boxplot(data=df, x="gene", y="expression", hue="condition", ax=ax)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=55)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, df


def plot_insilico_vs_observed_scatter(
    observed_adata,
    insilico_adata,
    genes: Sequence[str],
    *,
    layer: Optional[str] = None,
    figsize: Tuple[float, float] = (4.2, 4.2),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Scatter + regression of observed vs in-silico mean expression (Fig. 6e style).
    """
    if not genes:
        raise ValueError("genes must be a non-empty list.")
    obs_mean = []
    sim_mean = []
    for gene in genes:
        obs_mean.append(float(np.mean(_extract_gene_values(observed_adata, gene, layer=layer))))
        sim_mean.append(float(np.mean(_extract_gene_values(insilico_adata, gene, layer=layer))))
    x = np.asarray(obs_mean, dtype=float)
    y = np.asarray(sim_mean, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    corr, pval = pearsonr(x, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.scatter(x, y, s=28, color="#3C5488", alpha=0.8, edgecolor="white", linewidth=0.3)
    x_span = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(x_span, slope * x_span + intercept, color="#E64B35", linewidth=1.8)
    min_v = float(min(np.min(x), np.min(y)))
    max_v = float(max(np.max(x), np.max(y)))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1, color="grey", alpha=0.5)
    ax.set_xlabel("Observed mean expression")
    ax.set_ylabel("In-silico mean expression")
    ax.text(
        0.02,
        0.98,
        f"R={corr:.3f}\nR^2={r2:.3f}\np={pval:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, {"pearson_r": corr, "p_value": pval, "r2": r2}


def plot_distribution_overlap(
    observed: Sequence[float],
    insilico: Sequence[float],
    *,
    observed_label: str = "observed",
    insilico_label: str = "in_silico",
    bins: int = 60,
    kde: bool = True,
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 220,
    ax=None,
    save: Optional[str] = None,
    show: bool = True,
):
    """
    Plot distribution overlap for observed vs in-silico values (Fig. 5g/6f style).
    """
    obs = np.asarray(observed, dtype=float).reshape(-1)
    sim = np.asarray(insilico, dtype=float).reshape(-1)
    df = pd.DataFrame(
        {
            "value": np.concatenate([obs, sim]),
            "condition": [observed_label] * len(obs) + [insilico_label] * len(sim),
        }
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    sns.histplot(data=df, x="value", hue="condition", bins=bins, stat="density", common_norm=False, kde=kde, ax=ax, alpha=0.35)
    ax.set_ylabel("Density")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, df


__all__ = [
    "build_palette",
    "plot_stage_umap_grid",
    "plot_stage_cell_type_composition",
    "plot_marker_dotplot",
    "plot_dynamic_trajectory_graph",
    "extract_dynamic_marker_table",
    "plot_dynamic_marker_trajectories",
    "plot_dynamic_marker_heatmap",
    "plot_dynamic_marker_heatmap_track_panels",
    "extract_hierarchical_marker_table",
    "plot_hierarchical_marker_branch_heatmap",
    "plot_temporal_dynamics_graph",
    "extract_perturbation_table",
    "plot_perturbation_ranked_bar",
    "plot_gene_expression_shift",
    "plot_insilico_vs_observed_scatter",
    "plot_distribution_overlap",
]
