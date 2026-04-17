from __future__ import annotations

import json
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

import fcntl
import numpy as np
import pandas as pd
from anndata import read_h5ad
from UNAGI.palettes import DEFAULT_CATEGORICAL_COLORS

POINT_RENDER_THRESHOLD = 50_000
POINT_RETURN_THRESHOLD = 25_000
SAMPLED_POINT_LIMIT = 30_000
MAX_GRID_SIZE = 256
MIN_GRID_SIZE = 48
DEFAULT_SCREEN_WIDTH = 1200
DEFAULT_SCREEN_HEIGHT = 620
DEFAULT_HOT_STAGE_SHARDS = 4
DEFAULT_HOT_GENE_VECTORS = 8
EXPLORER_CACHE_VERSION = 3


def _cache_key(path: Path | None) -> tuple[str, int | None, int | None]:
    if path is None:
        return ("", None, None)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None, None)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _sorted_keys(values: Iterable[Any]) -> list[str]:
    def sort_key(value: Any) -> tuple[int, Any]:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(values, key=sort_key)]


def _resolve_stage_order(stage_series: pd.Series) -> list[str]:
    if isinstance(stage_series.dtype, pd.CategoricalDtype):
        present = set(stage_series.astype(str).tolist())
        return [str(value) for value in stage_series.dtype.categories.tolist() if str(value) in present]
    return [str(value) for value in pd.unique(stage_series.astype(str))]


def _resolve_string(options: Iterable[str], raw: Any, *, default: str | None = None) -> str:
    valid = [str(option) for option in options]
    if not valid:
        return default or ""
    text = str(raw) if raw not in (None, "") else ""
    if text in valid:
        return text
    return default if default in valid else valid[0]


def _build_gene_lookup(gene_names: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact_lookup = {name: name for name in gene_names}
    folded_lookup: dict[str, list[str]] = {}
    for name in gene_names:
        folded_lookup.setdefault(name.casefold(), []).append(name)
    return exact_lookup, folded_lookup


def _resolve_gene_name(gene_names: list[str], gene: str) -> str:
    raw = str(gene or "").strip()
    if not raw:
        return ""
    exact_lookup, folded_lookup = _build_gene_lookup(gene_names)
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


def _safe_stage_slug(stage: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(stage))
    token = token.strip("_") or "stage"
    digest = sha1(str(stage).encode("utf-8")).hexdigest()[:10]
    return f"{token[:24]}_{digest}"


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize_field_values(series: pd.Series) -> np.ndarray:
    values = series.astype(str).replace({"nan": "", "None": ""}).to_numpy(dtype=str)
    return values


def _default_color_by(color_options: list[str]) -> str:
    for field in ("ident", "leiden", "stage"):
        if field in color_options:
            return field
    return color_options[0] if color_options else ""


def _build_palette(categories: Iterable[str]) -> dict[str, str]:
    ordered = sorted({str(category) for category in categories}, key=_resolve_sort_key)
    return {
        category: DEFAULT_CATEGORICAL_COLORS[index % len(DEFAULT_CATEGORICAL_COLORS)]
        for index, category in enumerate(ordered)
    }


def _resolve_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def _grid_size(screen_width: int | None, screen_height: int | None) -> tuple[int, int]:
    width = int(screen_width or DEFAULT_SCREEN_WIDTH)
    height = int(screen_height or DEFAULT_SCREEN_HEIGHT)
    grid_x = int(np.clip(max(MIN_GRID_SIZE, width // 4), MIN_GRID_SIZE, MAX_GRID_SIZE))
    grid_y = int(np.clip(max(MIN_GRID_SIZE, height // 4), MIN_GRID_SIZE, MAX_GRID_SIZE))
    return grid_x, grid_y


def _stable_seed(*parts: Any) -> int:
    digest = sha1("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _open_backed_adata(dataset_path: Path):
    adata = read_h5ad(dataset_path, backed="r")
    try:
        yield adata
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


@dataclass
class StageShard:
    stage: str
    cell_ids: np.ndarray
    coords: np.ndarray
    field_codes: dict[str, np.ndarray]
    field_categories: dict[str, list[str]]
    bounds: dict[str, float]


class UmapViewportCache:
    def __init__(
        self,
        *,
        skill_root: Path | None = None,
        hot_stage_limit: int = DEFAULT_HOT_STAGE_SHARDS,
        hot_gene_limit: int = DEFAULT_HOT_GENE_VECTORS,
    ) -> None:
        self.skill_root = Path(skill_root).expanduser().resolve() if skill_root else None
        if self.skill_root is not None:
            self.cache_root = self.skill_root / "outputs" / "explorer_cache"
        else:
            self.cache_root = Path(tempfile.gettempdir()) / "unagi_explorer_cache"
        self.hot_stage_limit = int(hot_stage_limit)
        self.hot_gene_limit = int(hot_gene_limit)
        self._lock = threading.Lock()
        self._manifest_lru: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stage_lru: OrderedDict[tuple[str, str], StageShard] = OrderedDict()
        self._gene_lru: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()

    def get_payload(
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
        analysis_path = Path(analysis_dir).expanduser().resolve()
        manifest = self.get_manifest(analysis_path)
        stage_options = manifest["stages"]
        selected_stage = _resolve_string(stage_options, stage, default=stage_options[0] if stage_options else "")
        selected_color = _resolve_string(manifest["color_options"], color_by, default=manifest["default_color_by"])
        selected_gene = _resolve_gene_name(manifest["gene_names"], gene) if str(gene or "").strip() else ""
        stage_shard = self.get_stage_shard(analysis_path, stage=selected_stage)
        viewport = self._normalize_viewport(
            stage_shard.bounds,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        visible_mask = self._viewport_mask(stage_shard.coords, viewport)
        visible_count = int(visible_mask.sum())
        render_mode, render_reason = self._decide_render_mode(
            visible_count=visible_count,
            current_render_mode=current_render_mode,
            has_gene=bool(selected_gene),
        )

        payload: dict[str, Any] = {
            "render_mode": render_mode,
            "render_reason": render_reason,
            "stage": selected_stage,
            "stage_options": stage_options,
            "color_by": selected_color,
            "color_options": manifest["color_options"],
            "color_palette": manifest.get("field_palettes", {}).get(selected_color, {}),
            "default_color_by": manifest["default_color_by"],
            "color_mode": "gene" if selected_gene else "metadata",
            "expression_gene": selected_gene or None,
            "visible_cell_count": visible_count,
            "stage_cell_count": int(manifest["stage_info"][selected_stage]["cell_count"]),
            "selection_supported": render_mode == "points",
            "thresholds": {
                "point_mode_max_visible": POINT_RENDER_THRESHOLD,
                "density_mode_return_visible": POINT_RETURN_THRESHOLD,
                "sampled_point_limit": SAMPLED_POINT_LIMIT,
            },
            "stage_bounds": stage_shard.bounds,
            "viewport": viewport,
            "point_count": 0,
            "density": None,
            "points": [],
        }

        visible_indices = np.flatnonzero(visible_mask)
        if render_mode == "points":
            payload["point_count"] = int(len(visible_indices))
            payload["points"] = self._point_payload(
                stage_shard,
                visible_indices=visible_indices,
                color_by=selected_color,
                gene=selected_gene,
                analysis_path=analysis_path,
            )
            return payload

        if render_mode == "sampled_points":
            sampled_indices = self._sample_visible_indices(
                stage_shard,
                visible_indices=visible_indices,
                color_by=selected_color,
                analysis_path=analysis_path,
            )
            payload["point_count"] = int(len(sampled_indices))
            payload["selection_supported"] = False
            payload["points"] = self._point_payload(
                stage_shard,
                visible_indices=sampled_indices,
                color_by=selected_color,
                gene="",
                analysis_path=analysis_path,
            )
            return payload

        payload["density"] = self._density_payload(
            stage_shard,
            visible_indices=visible_indices,
            gene=selected_gene,
            analysis_path=analysis_path,
            viewport=viewport,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        return payload

    def summarize_selection(
        self,
        analysis_dir: str | Path,
        *,
        stage: str,
        color_by: str,
        cell_ids: list[int],
        gene: str | None = None,
        render_mode: str | None = None,
    ) -> dict[str, Any]:
        if str(render_mode or "points") != "points":
            raise ValueError("Selection summaries are available only in point mode.")
        if not cell_ids:
            return {
                "summary_type": "empty",
                "selected_count": 0,
                "columns": [],
                "rows": [],
                "message": "No cells selected.",
            }
        analysis_path = Path(analysis_dir).expanduser().resolve()
        manifest = self.get_manifest(analysis_path)
        selected_stage = _resolve_string(manifest["stages"], stage, default=manifest["stages"][0] if manifest["stages"] else "")
        selected_color = _resolve_string(manifest["color_options"], color_by, default=manifest["default_color_by"])
        selected_gene = _resolve_gene_name(manifest["gene_names"], gene) if str(gene or "").strip() else ""
        stage_shard = self.get_stage_shard(analysis_path, stage=selected_stage)
        stage_cell_ids = stage_shard.cell_ids
        requested = np.asarray(sorted(set(int(cell_id) for cell_id in cell_ids)), dtype=np.int64)
        positions = np.searchsorted(stage_cell_ids, requested)
        valid = positions < len(stage_cell_ids)
        positions = positions[valid]
        requested = requested[valid]
        matches = stage_cell_ids[positions] == requested
        positions = positions[matches]
        if len(positions) == 0:
            return {
                "summary_type": "empty",
                "selected_count": 0,
                "columns": [],
                "rows": [],
                "message": "No selected cells matched the current stage.",
            }

        if selected_gene:
            values = self.get_gene_vector(analysis_path, stage=selected_stage, gene=selected_gene)[positions]
            return {
                "summary_type": "gene",
                "selected_count": int(len(positions)),
                "columns": ["gene", "selected_cells", "mean", "min", "max"],
                "rows": [[
                    selected_gene,
                    str(len(positions)),
                    f"{float(np.mean(values)):.4f}",
                    f"{float(np.min(values)):.4f}",
                    f"{float(np.max(values)):.4f}",
                ]],
                "message": f"Selected {len(positions)} cells.",
            }

        codes = stage_shard.field_codes[selected_color][positions]
        categories = stage_shard.field_categories[selected_color]
        counts = np.bincount(codes, minlength=len(categories))
        rows = [
            [categories[index], str(int(count))]
            for index, count in enumerate(counts)
            if int(count) > 0
        ]
        rows.sort(key=lambda item: int(item[1]), reverse=True)
        return {
            "summary_type": "metadata",
            "selected_count": int(len(positions)),
            "columns": [selected_color, "count"],
            "rows": rows,
            "message": f"Selected {len(positions)} cells.",
        }

    def get_manifest(self, analysis_dir: str | Path) -> dict[str, Any]:
        analysis_path = Path(analysis_dir).expanduser().resolve()
        dataset_path = analysis_path / "dataset.h5ad"
        attribute_path = analysis_path / "attribute.pkl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Analysis dataset not found: {dataset_path}")
        cache_key = _hash_payload(
            {
                "version": EXPLORER_CACHE_VERSION,
                "analysis_dir": str(analysis_path),
                "dataset": _cache_key(dataset_path),
                "attribute": _cache_key(attribute_path if attribute_path.exists() else None),
            }
        )
        with self._lock:
            cached = self._manifest_lru.get(cache_key)
            if cached is not None:
                self._manifest_lru.move_to_end(cache_key)
                return cached

        cache_dir = self.cache_root / cache_key
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            lock_path = cache_dir / ".manifest.lock"
            with _file_lock(lock_path):
                if not manifest_path.exists():
                    manifest = self._build_manifest(
                        analysis_path=analysis_path,
                        dataset_path=dataset_path,
                        attribute_path=attribute_path if attribute_path.exists() else None,
                        cache_dir=cache_dir,
                    )
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cache_dir"] = str(cache_dir)
        manifest["analysis_hash"] = cache_key
        with self._lock:
            self._manifest_lru[cache_key] = manifest
            while len(self._manifest_lru) > self.hot_stage_limit:
                self._manifest_lru.popitem(last=False)
        return manifest

    def get_stage_shard(self, analysis_dir: str | Path, *, stage: str) -> StageShard:
        analysis_path = Path(analysis_dir).expanduser().resolve()
        manifest = self.get_manifest(analysis_path)
        stage_map = manifest["stage_info"]
        if stage not in stage_map:
            raise KeyError(f"Unknown stage: {stage}")
        cache_key = (manifest["analysis_hash"], stage)
        with self._lock:
            cached = self._stage_lru.get(cache_key)
            if cached is not None:
                self._stage_lru.move_to_end(cache_key)
                return cached

        stage_entry = stage_map[stage]
        cache_dir = Path(manifest["cache_dir"])
        stage_dir = cache_dir / "stages" / stage_entry["slug"]
        shard_path = stage_dir / "stage_shard.npz"
        meta_path = stage_dir / "stage_meta.json"
        if not shard_path.exists() or not meta_path.exists():
            lock_path = stage_dir / ".stage.lock"
            with _file_lock(lock_path):
                if not shard_path.exists() or not meta_path.exists():
                    self._build_stage_shard(analysis_path, manifest, stage=stage, stage_dir=stage_dir)

        shard_npz = np.load(shard_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        field_codes = {
            field: shard_npz[f"field_{field_index}_codes"]
            for field, field_index in meta["field_index"].items()
        }
        shard = StageShard(
            stage=stage,
            cell_ids=shard_npz["cell_ids"].astype(np.int64, copy=False),
            coords=shard_npz["coords"].astype(np.float32, copy=False),
            field_codes=field_codes,
            field_categories={field: [str(value) for value in values] for field, values in meta["field_categories"].items()},
            bounds={key: float(value) for key, value in meta["bounds"].items()},
        )
        with self._lock:
            self._stage_lru[cache_key] = shard
            while len(self._stage_lru) > self.hot_stage_limit:
                self._stage_lru.popitem(last=False)
        return shard

    def get_gene_vector(self, analysis_dir: str | Path, *, stage: str, gene: str) -> np.ndarray:
        analysis_path = Path(analysis_dir).expanduser().resolve()
        manifest = self.get_manifest(analysis_path)
        canonical_gene = _resolve_gene_name(manifest["gene_names"], gene)
        cache_key = (manifest["analysis_hash"], stage, canonical_gene)
        with self._lock:
            cached = self._gene_lru.get(cache_key)
            if cached is not None:
                self._gene_lru.move_to_end(cache_key)
                return cached

        stage_entry = manifest["stage_info"][stage]
        cache_dir = Path(manifest["cache_dir"])
        gene_dir = cache_dir / "genes" / stage_entry["slug"]
        gene_name_token = "".join(char if char.isalnum() else "_" for char in canonical_gene)[:32] or "gene"
        gene_path = gene_dir / f"{gene_name_token}_{sha1(canonical_gene.encode('utf-8')).hexdigest()[:10]}.npy"
        if not gene_path.exists():
            lock_path = gene_dir / ".gene.lock"
            with _file_lock(lock_path):
                if not gene_path.exists():
                    self._build_gene_vector(
                        analysis_path=analysis_path,
                        manifest=manifest,
                        stage=stage,
                        gene=canonical_gene,
                        gene_path=gene_path,
                    )
        values = np.load(gene_path, allow_pickle=False).astype(np.float32, copy=False)
        with self._lock:
            self._gene_lru[cache_key] = values
            while len(self._gene_lru) > self.hot_gene_limit:
                self._gene_lru.popitem(last=False)
        return values

    def _build_manifest(
        self,
        *,
        analysis_path: Path,
        dataset_path: Path,
        attribute_path: Path | None,
        cache_dir: Path,
    ) -> dict[str, Any]:
        with _open_backed_adata(dataset_path) as adata:
            embedding_key = "X_umap" if "X_umap" in adata.obsm else "umap" if "umap" in adata.obsm else None
            if embedding_key is None:
                raise KeyError("No UMAP embedding is available. Expected adata.obsm['X_umap'] or adata.obsm['umap'].")
            obs = adata.obs
            if "stage" not in obs:
                raise KeyError("Missing stage key in adata.obs: stage")
            stage_series = obs["stage"]
            stages = _resolve_stage_order(stage_series)
            coords = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(f"UMAP embedding must have shape (n_cells, 2+); got {coords.shape}")
            color_options: list[str] = []
            preferred_color_fields = ["stage", "ident", "leiden", "name.simple", "final.ident", "Sample.ID"]
            seen_fields: set[str] = set()
            for field in preferred_color_fields + [str(column) for column in obs.columns]:
                if field in seen_fields or field not in obs:
                    continue
                seen_fields.add(field)
                series = obs[field]
                if series.isna().all():
                    continue
                unique_count = int(series.astype(str).nunique())
                if unique_count <= 1:
                    continue
                if unique_count <= 80 or field in {"stage", "ident", "leiden"}:
                    color_options.append(field)
            field_categories: dict[str, list[str]] = {}
            field_palettes: dict[str, dict[str, str]] = {}
            stage_info: dict[str, Any] = {}
            stage_values = stage_series.astype(str).to_numpy()
            for field in color_options:
                series = obs[field].astype(str).replace({"nan": "", "None": ""})
                categories = sorted(series.unique().tolist(), key=_resolve_sort_key)
                field_categories[field] = [str(value) for value in categories]
                field_palettes[field] = _build_palette(field_categories[field])
            for stage in stages:
                mask = stage_values == str(stage)
                stage_coords = coords[mask]
                if len(stage_coords) == 0:
                    continue
                stage_info[str(stage)] = {
                    "slug": _safe_stage_slug(str(stage)),
                    "cell_count": int(mask.sum()),
                    "bounds": {
                        "x_min": float(np.min(stage_coords[:, 0])),
                        "x_max": float(np.max(stage_coords[:, 0])),
                        "y_min": float(np.min(stage_coords[:, 1])),
                        "y_max": float(np.max(stage_coords[:, 1])),
                    },
                }
            gene_names = [str(name) for name in adata.var_names.astype(str).tolist()]

        return {
            "analysis_dir": str(analysis_path),
            "dataset_path": str(dataset_path),
            "attribute_path": str(attribute_path) if attribute_path else None,
            "embedding_key": embedding_key,
            "cache_dir": str(cache_dir),
            "stages": stages,
            "color_options": color_options,
            "field_categories": field_categories,
            "field_palettes": field_palettes,
            "default_color_by": _default_color_by(color_options),
            "gene_names": gene_names,
            "stage_info": stage_info,
            "thresholds": {
                "point_mode_max_visible": POINT_RENDER_THRESHOLD,
                "density_mode_return_visible": POINT_RETURN_THRESHOLD,
                "max_grid_size": MAX_GRID_SIZE,
            },
        }

    def _build_stage_shard(self, analysis_path: Path, manifest: dict[str, Any], *, stage: str, stage_dir: Path) -> None:
        dataset_path = Path(manifest["dataset_path"])
        with _open_backed_adata(dataset_path) as adata:
            obs = adata.obs
            stage_values = obs["stage"].astype(str).to_numpy()
            mask = stage_values == str(stage)
            if not mask.any():
                raise ValueError(f"No cells were found for stage={stage!r}.")
            coords = np.asarray(adata.obsm[manifest["embedding_key"]], dtype=np.float32)[mask]
            cell_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
            field_categories: dict[str, list[str]] = {}
            field_index: dict[str, int] = {}
            payload: dict[str, np.ndarray] = {
                "coords": coords,
                "cell_ids": cell_ids,
            }
            for field_idx, field in enumerate(manifest["color_options"]):
                if field not in obs:
                    continue
                values = _normalize_field_values(obs.loc[mask, field])
                categories, inverse = np.unique(values, return_inverse=True)
                field_categories[field] = [str(value) for value in categories.tolist()]
                field_index[field] = field_idx
                payload[f"field_{field_idx}_codes"] = inverse.astype(np.int32, copy=False)
            stage_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(stage_dir / "stage_shard.npz", **payload)
            meta = {
                "stage": str(stage),
                "cell_count": int(len(cell_ids)),
                "bounds": manifest["stage_info"][stage]["bounds"],
                "field_index": field_index,
                "field_categories": field_categories,
            }
            (stage_dir / "stage_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _build_gene_vector(
        self,
        *,
        analysis_path: Path,
        manifest: dict[str, Any],
        stage: str,
        gene: str,
        gene_path: Path,
    ) -> None:
        stage_shard = self.get_stage_shard(analysis_path, stage=stage)
        dataset_path = Path(manifest["dataset_path"])
        with _open_backed_adata(dataset_path) as adata:
            gene_view = adata[stage_shard.cell_ids, gene]
            matrix = gene_view.X
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            array = np.asarray(matrix, dtype=np.float32).reshape(len(stage_shard.cell_ids))
        gene_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gene_path, array, allow_pickle=False)

    def _normalize_viewport(
        self,
        stage_bounds: dict[str, float],
        *,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
    ) -> dict[str, float | bool]:
        min_x = float(stage_bounds["x_min"])
        max_x = float(stage_bounds["x_max"])
        min_y = float(stage_bounds["y_min"])
        max_y = float(stage_bounds["y_max"])
        if None in {x_min, x_max, y_min, y_max}:
            return {
                "x_min": min_x,
                "x_max": max_x,
                "y_min": min_y,
                "y_max": max_y,
                "is_full_stage": True,
            }
        clipped = {
            "x_min": float(max(min_x, min(float(x_min), max_x))),
            "x_max": float(max(min_x, min(float(x_max), max_x))),
            "y_min": float(max(min_y, min(float(y_min), max_y))),
            "y_max": float(max(min_y, min(float(y_max), max_y))),
        }
        if clipped["x_min"] > clipped["x_max"]:
            clipped["x_min"], clipped["x_max"] = clipped["x_max"], clipped["x_min"]
        if clipped["y_min"] > clipped["y_max"]:
            clipped["y_min"], clipped["y_max"] = clipped["y_max"], clipped["y_min"]
        if np.isclose(clipped["x_min"], clipped["x_max"]):
            pad_x = max((max_x - min_x) * 0.005, 1e-6)
            clipped["x_min"] -= pad_x
            clipped["x_max"] += pad_x
        if np.isclose(clipped["y_min"], clipped["y_max"]):
            pad_y = max((max_y - min_y) * 0.005, 1e-6)
            clipped["y_min"] -= pad_y
            clipped["y_max"] += pad_y
        clipped["is_full_stage"] = bool(
            np.isclose(clipped["x_min"], min_x)
            and np.isclose(clipped["x_max"], max_x)
            and np.isclose(clipped["y_min"], min_y)
            and np.isclose(clipped["y_max"], max_y)
        )
        return clipped

    def _viewport_mask(self, coords: np.ndarray, viewport: dict[str, float | bool]) -> np.ndarray:
        return (
            (coords[:, 0] >= float(viewport["x_min"]))
            & (coords[:, 0] <= float(viewport["x_max"]))
            & (coords[:, 1] >= float(viewport["y_min"]))
            & (coords[:, 1] <= float(viewport["y_max"]))
        )

    def _decide_render_mode(
        self,
        *,
        visible_count: int,
        current_render_mode: str | None,
        has_gene: bool,
    ) -> tuple[str, str]:
        current_mode = str(current_render_mode or "")
        if has_gene:
            if current_mode == "density":
                if visible_count <= POINT_RETURN_THRESHOLD:
                    return (
                        "points",
                        f"Zoomed to {visible_count} cells; returning to exact points.",
                    )
                return (
                    "density",
                    f"Showing density for {visible_count} visible cells.",
                )
            if visible_count > POINT_RENDER_THRESHOLD:
                return (
                    "density",
                    f"{visible_count} visible cells exceed the {POINT_RENDER_THRESHOLD:,} point threshold.",
                )
            return (
                "points",
                f"Showing {visible_count} exact cells.",
            )
        if visible_count > POINT_RENDER_THRESHOLD:
            return (
                "sampled_points",
                f"{visible_count} visible cells exceed the {POINT_RENDER_THRESHOLD:,} point threshold; showing sampled colored points.",
            )
        return (
            "points",
            f"Showing {visible_count} exact cells.",
        )

    def _sample_visible_indices(
        self,
        stage_shard: StageShard,
        *,
        visible_indices: np.ndarray,
        color_by: str,
        analysis_path: Path,
    ) -> np.ndarray:
        if len(visible_indices) <= SAMPLED_POINT_LIMIT:
            return visible_indices

        visible_codes = stage_shard.field_codes[color_by][visible_indices]
        categories, counts = np.unique(visible_codes, return_counts=True)
        desired = np.floor((counts / counts.sum()) * SAMPLED_POINT_LIMIT).astype(int)
        desired = np.maximum(desired, 1)
        desired = np.minimum(desired, counts)

        total = int(desired.sum())
        if total > SAMPLED_POINT_LIMIT:
            overflow = total - SAMPLED_POINT_LIMIT
            order = np.argsort(-desired)
            for index in order:
                if overflow <= 0:
                    break
                reducible = max(0, int(desired[index]) - 1)
                if reducible <= 0:
                    continue
                delta = min(reducible, overflow)
                desired[index] -= delta
                overflow -= delta
        elif total < SAMPLED_POINT_LIMIT:
            remainder = SAMPLED_POINT_LIMIT - total
            order = np.argsort(-(counts - desired))
            for index in order:
                if remainder <= 0:
                    break
                spare = int(counts[index] - desired[index])
                if spare <= 0:
                    continue
                delta = min(spare, remainder)
                desired[index] += delta
                remainder -= delta

        rng = np.random.default_rng(
            _stable_seed(analysis_path, stage_shard.stage, color_by, len(visible_indices))
        )
        sampled_chunks: list[np.ndarray] = []
        for category_code, target in zip(categories, desired):
            category_indices = visible_indices[visible_codes == category_code]
            target = int(target)
            if len(category_indices) <= target:
                sampled_chunks.append(category_indices)
                continue
            sampled = rng.choice(category_indices, size=target, replace=False)
            sampled_chunks.append(np.sort(sampled.astype(np.int64, copy=False)))
        if not sampled_chunks:
            return visible_indices[:0]
        return np.sort(np.concatenate(sampled_chunks))

    def _point_payload(
        self,
        stage_shard: StageShard,
        *,
        visible_indices: np.ndarray,
        color_by: str,
        gene: str,
        analysis_path: Path,
    ) -> list[dict[str, Any]]:
        coords = stage_shard.coords[visible_indices]
        cell_ids = stage_shard.cell_ids[visible_indices]
        points: list[dict[str, Any]] = []
        if gene:
            expression = self.get_gene_vector(analysis_path, stage=stage_shard.stage, gene=gene)[visible_indices]
            for index, coord in enumerate(coords):
                points.append(
                    {
                        "cell_id": int(cell_ids[index]),
                        "x": float(coord[0]),
                        "y": float(coord[1]),
                        "expression": float(expression[index]),
                    }
                )
            return points

        categories = stage_shard.field_categories[color_by]
        codes = stage_shard.field_codes[color_by][visible_indices]
        for index, coord in enumerate(coords):
            points.append(
                {
                    "cell_id": int(cell_ids[index]),
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "color_value": categories[int(codes[index])],
                }
            )
        return points

    def _density_payload(
        self,
        stage_shard: StageShard,
        *,
        visible_indices: np.ndarray,
        gene: str,
        analysis_path: Path,
        viewport: dict[str, float | bool],
        screen_width: int | None,
        screen_height: int | None,
    ) -> dict[str, Any]:
        visible_coords = stage_shard.coords[visible_indices]
        grid_x, grid_y = _grid_size(screen_width, screen_height)
        range_x = [float(viewport["x_min"]), float(viewport["x_max"])]
        range_y = [float(viewport["y_min"]), float(viewport["y_max"])]
        counts, x_edges, y_edges = np.histogram2d(
            visible_coords[:, 0],
            visible_coords[:, 1],
            bins=[grid_x, grid_y],
            range=[range_x, range_y],
        )
        counts = counts.astype(np.float32, copy=False)
        x_centers = ((x_edges[:-1] + x_edges[1:]) / 2.0).astype(np.float32, copy=False)
        y_centers = ((y_edges[:-1] + y_edges[1:]) / 2.0).astype(np.float32, copy=False)
        if gene:
            values = self.get_gene_vector(analysis_path, stage=stage_shard.stage, gene=gene)[visible_indices]
            weighted, _x_edges, _y_edges = np.histogram2d(
                visible_coords[:, 0],
                visible_coords[:, 1],
                bins=[grid_x, grid_y],
                range=[range_x, range_y],
                weights=values,
            )
            mean_values = np.full_like(weighted, np.nan, dtype=np.float32)
            np.divide(
                weighted,
                counts,
                out=mean_values,
                where=counts > 0,
            )
            z_payload = [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in mean_values.T
            ]
            return {
                "metric": "mean_expression",
                "gene": gene,
                "x": x_centers.tolist(),
                "y": y_centers.tolist(),
                "z": z_payload,
                "counts": counts.T.astype(np.int32, copy=False).tolist(),
                "grid_shape": [int(grid_y), int(grid_x)],
            }

        return {
            "metric": "count",
            "gene": None,
            "x": x_centers.tolist(),
            "y": y_centers.tolist(),
            "z": counts.T.tolist(),
            "counts": counts.T.astype(np.int32, copy=False).tolist(),
            "grid_shape": [int(grid_y), int(grid_x)],
        }
