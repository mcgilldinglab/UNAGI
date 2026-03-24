from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

SUMMARY_DIR_NAME = "idrem_track_summaries"


def _cache_key(path: Path | None) -> tuple[str, int | None, int | None]:
    if path is None:
        return ("", None, None)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None, None)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _letter_label(index: int) -> str:
    value = int(index)
    if value < 0:
        raise ValueError("Path index must be non-negative.")
    label = ""
    while True:
        value, rem = divmod(value, 26)
        label = chr(65 + rem) + label
        if value == 0:
            break
        value -= 1
    return label


def _normalize_drem_json(text: str) -> str:
    payload = text.strip()
    if payload.startswith("data="):
        payload = payload[5:]
    if payload.endswith(";"):
        payload = payload[:-1]
    payload = re.sub(r",\s*]", "]", payload)
    payload = re.sub(r",\s*}", "}", payload)
    return payload


def _parse_numeric_text(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _stage_order(nodetime: Any) -> int:
    text = str(nodetime)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return 0


def _load_drem_bundle(track_dir: Path) -> list[Any]:
    bundle_path = track_dir / "DREM.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing DREM.json in {track_dir}")
    payload = _normalize_drem_json(bundle_path.read_text(encoding="utf-8"))
    return json.loads(payload)


def _node_genes(gene_index_flags: list[Any] | None, exgene: list[str]) -> list[str]:
    if not isinstance(gene_index_flags, list):
        return []
    out: list[str] = []
    for index, flag in enumerate(gene_index_flags):
        if flag and index < len(exgene):
            out.append(str(exgene[index]))
    return out


def _build_gene_expression_lookup(geneabex: list[Any] | None) -> tuple[list[str], dict[str, list[str]]]:
    if not isinstance(geneabex, list) or not geneabex:
        return [], {}
    header = geneabex[0]
    if not isinstance(header, list) or len(header) < 2:
        return [], {}
    stages = [str(item) for item in header[1:]]
    rows: dict[str, list[str]] = {}
    for raw in geneabex[1:]:
        if not isinstance(raw, list) or not raw:
            continue
        gene = str(raw[0])
        rows[gene] = [str(item) for item in raw[1:]]
    return stages, rows


def _path_regulators(path_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tf_scores: dict[str, float] = {}
    mir_scores: dict[str, float] = {}
    for node in path_nodes:
        if len(node.get("path_labels", [])) != 1:
            continue
        for tf_entry in node.get("ETF", []) or []:
            if not isinstance(tf_entry, list) or not tf_entry:
                continue
            tf_name = str(tf_entry[0]).split(" ")[0].strip()
            if not tf_name:
                continue
            score_idx = 5 if len(tf_entry) < 7 else 6
            score = _parse_numeric_text(tf_entry[score_idx] if score_idx < len(tf_entry) else None)
            if score is None:
                continue
            bucket = mir_scores if tf_name.upper().startswith(("MIR-", "LET-")) else tf_scores
            if tf_name in bucket:
                bucket[tf_name] = max(bucket[tf_name], score)
            else:
                bucket[tf_name] = score

    rows: list[dict[str, Any]] = []
    for kind, bucket, limit in (("TF", tf_scores, 20), ("miRNA", mir_scores, 10)):
        for name, score in sorted(bucket.items(), key=lambda item: (item[1], item[0]))[:limit]:
            rows.append(
                {
                    "Regulator": name,
                    "Type": kind,
                    "Score": round(float(score), 6),
                }
            )
    return rows


def _node_top_tfs(node: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tf_entry in node.get("ETF", []) or []:
        if not isinstance(tf_entry, list) or not tf_entry:
            continue
        tf_name = str(tf_entry[0]).split(" ")[0].strip()
        if not tf_name or tf_name.upper().startswith(("MIR-", "LET-")):
            continue
        score_idx = 5 if len(tf_entry) < 7 else 6
        score = _parse_numeric_text(tf_entry[score_idx] if score_idx < len(tf_entry) else None)
        rows.append({"name": tf_name, "score": score})
    rows.sort(key=lambda item: ((item["score"] if item["score"] is not None else float("inf")), item["name"]))
    return rows[:limit]


def _path_enrichment(leaf: dict[str, Any]) -> list[dict[str, Any]]:
    def _format_pvalue(value: Any) -> str:
        numeric = _parse_numeric_text(value)
        if numeric is None:
            return ""
        return f"{numeric:.4e}"

    rows: list[dict[str, Any]] = []
    for raw in leaf.get("goNode", []) or []:
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        corrected_numeric = _parse_numeric_text(raw[3] if len(raw) > 3 else None)
        if corrected_numeric is None or corrected_numeric >= 0.05:
            continue
        rows.append(
            {
                "GO Term ID": str(raw[0]),
                "GO Term Name": str(raw[1]),
                "p-value": _format_pvalue(raw[2] if len(raw) > 2 else None),
                "corrected p-value": _format_pvalue(corrected_numeric),
            }
        )
    return rows


def _empty_enrichment_table(message: str = "No enrichment rows were available for this path.") -> dict[str, Any]:
    return {
        "columns": ["Category", "Term ID", "Term Name", "corrected p-value", "p-value", "Genes in query"],
        "rows": [],
        "row_count": 0,
        "hover_texts": [],
        "message": message,
        "source": "ToppGene",
    }


def _format_scientific(value: Any) -> str:
    numeric = _parse_numeric_text(value)
    if numeric is None:
        return ""
    return f"{numeric:.4e}"


def _normalize_symbols(genes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for gene in genes:
        symbol = str(gene or "").strip()
        if not symbol:
            continue
        key = symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(symbol)
    return tuple(out)


def _overlap_gene_preview(genes: list[str], *, limit: int = 12) -> str:
    if not genes:
        return ""
    if len(genes) <= limit:
        return ", ".join(genes)
    return ", ".join(genes[:limit]) + f", +{len(genes) - limit} more"


def _summary_cache_dir(analysis_dir: Path) -> Path:
    return analysis_dir / SUMMARY_DIR_NAME


def _summary_cache_path(analysis_dir: Path, track_key: str) -> Path:
    return _summary_cache_dir(analysis_dir) / f"{track_key}.json"


def _track_fingerprint(analysis_path: Path, track_dir: Path) -> dict[str, Any]:
    return {
        "analysis_dir": str(analysis_path.resolve()),
        "track_dir": str(track_dir.resolve()),
        "drem_json": list(_cache_key(track_dir / "DREM.json")),
        "idrem_result_html": list(_cache_key(track_dir / "idrem_result.html")),
    }


def _load_persisted_summary(summary_path: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("source_fingerprint") != fingerprint:
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def _persist_summary(summary_path: Path, fingerprint: dict[str, Any], summary: dict[str, Any]) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "source_fingerprint": fingerprint,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


class ToppGeneEnrichmentClient:
    LOOKUP_URL = "https://toppgene.cchmc.org/API/lookup?pretty=true"
    ENRICH_URL = "https://toppgene.cchmc.org/API/enrich?pretty=true"
    CATEGORY_MAP = {
        "GeneOntologyBiologicalProcess": "GO Biological Process",
        "Pathway": "Pathway",
    }

    def __init__(self, *, timeout: int = 30) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "UNAGI/1.0",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(f"ToppGene request failed: {exc}") from exc

    def _lookup_entrez_ids(self, genes: tuple[str, ...]) -> list[int]:
        lookup = self._post_json(self.LOOKUP_URL, {"Symbols": list(genes)})
        genes_payload = lookup.get("Genes", []) if isinstance(lookup, dict) else []
        entrez_ids: list[int] = []
        seen: set[int] = set()
        for gene in genes_payload:
            if not isinstance(gene, dict) or "Entrez" not in gene:
                continue
            try:
                entrez_id = int(gene["Entrez"])
            except (TypeError, ValueError):
                continue
            if entrez_id in seen:
                continue
            seen.add(entrez_id)
            entrez_ids.append(entrez_id)
        return entrez_ids

    def _build_enrichment_table(self, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        columns = ["Category", "Term ID", "Term Name", "corrected p-value", "p-value", "Genes in query"]
        rows: list[list[str]] = []
        hover_texts: list[str] = []
        filtered_annotations: list[dict[str, Any]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            category_key = str(annotation.get("Category") or "")
            if category_key not in self.CATEGORY_MAP:
                continue
            corrected = _parse_numeric_text(annotation.get("QValueFDRBH"))
            if corrected is None or corrected >= 0.05:
                continue
            filtered_annotations.append(annotation)
        filtered_annotations.sort(
            key=lambda item: (
                _parse_numeric_text(item.get("QValueFDRBH")) if _parse_numeric_text(item.get("QValueFDRBH")) is not None else float("inf"),
                str(item.get("Category") or ""),
                str(item.get("Name") or ""),
            )
        )
        for annotation in filtered_annotations:
            overlap_genes = [
                str(gene.get("Symbol") or "").strip()
                for gene in annotation.get("Genes", []) or []
                if isinstance(gene, dict) and str(gene.get("Symbol") or "").strip()
            ]
            rows.append(
                [
                    self.CATEGORY_MAP[str(annotation.get("Category") or "")],
                    str(annotation.get("ID") or ""),
                    str(annotation.get("Name") or ""),
                    _format_scientific(annotation.get("QValueFDRBH")),
                    _format_scientific(annotation.get("PValue")),
                    _overlap_gene_preview(overlap_genes),
                ]
            )
            if overlap_genes:
                hover_texts.append(
                    "Matched genes: "
                    + ", ".join(overlap_genes)
                )
            else:
                hover_texts.append("Matched genes unavailable")
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "hover_texts": hover_texts,
            "message": "No enrichment rows with corrected p-value < 0.05 were available from ToppGene." if not rows else "",
            "source": "ToppGene",
        }

    def fetch(self, genes: list[str] | tuple[str, ...]) -> dict[str, Any]:
        normalized = _normalize_symbols(genes)
        if not normalized:
            return _empty_enrichment_table("No genes were available for ToppGene enrichment.")
        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
        entrez_ids = self._lookup_entrez_ids(normalized)
        if not entrez_ids:
            table = _empty_enrichment_table("ToppGene could not resolve the path genes to Entrez IDs.")
        else:
            enrich_payload = {
                "Genes": entrez_ids,
                "Categories": [
                    {
                        "Type": "GeneOntologyBiologicalProcess",
                        "PValue": 0.05,
                        "MinGenes": 1,
                        "MaxGenes": 2000,
                        "MaxResults": 200,
                        "Correction": "FDR",
                    },
                    {
                        "Type": "Pathway",
                        "PValue": 0.05,
                        "MinGenes": 1,
                        "MaxGenes": 2000,
                        "MaxResults": 200,
                        "Correction": "FDR",
                    },
                ],
            }
            enrich = self._post_json(self.ENRICH_URL, enrich_payload)
            annotations = enrich.get("Annotations", []) if isinstance(enrich, dict) else []
            table = self._build_enrichment_table(annotations)
        with self._lock:
            self._cache[normalized] = table
        return table


def _path_gene_rows(
    leaf: dict[str, Any],
    *,
    gene_expression_stages: list[str],
    gene_expression_lookup: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gene in leaf.get("genes", []):
        row = {"Gene": str(gene)}
        expression_values = gene_expression_lookup.get(str(gene))
        if expression_values:
            for stage_name, value in zip(gene_expression_stages, expression_values):
                row[str(stage_name)] = value
        rows.append(row)
    return rows


def _to_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"columns": [], "rows": [], "row_count": 0}
    columns = list(rows[0].keys())
    table_rows = [[str(row.get(column, "")) for column in columns] for row in rows]
    return {"columns": columns, "rows": table_rows, "row_count": len(rows)}


def _resolve_idrem_track_dir(analysis_dir: Path, track_key: str) -> Path:
    idrem_dir = analysis_dir / "idrem"
    if not idrem_dir.exists():
        raise FileNotFoundError(f"Missing idrem directory in {analysis_dir}")
    candidate = idrem_dir / f"{track_key}.txt_viz"
    if candidate.exists():
        return candidate
    matches = sorted(idrem_dir.glob(f"{track_key}*.txt_viz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Track {track_key!r} does not exist under {idrem_dir}")


def _build_summary(
    track_dir: Path,
    *,
    track_key: str,
    track_label: str,
    enrichment_client: ToppGeneEnrichmentClient,
) -> dict[str, Any]:
    payload = _load_drem_bundle(track_dir)
    raw_nodes = payload[0] if len(payload) > 0 else []
    exgene = [str(value) for value in (payload[3] if len(payload) > 3 and isinstance(payload[3], list) else [])]
    geneabex = payload[8] if len(payload) > 8 else None
    gene_expression_stages, gene_expression_lookup = _build_gene_expression_lookup(geneabex)

    nodes_by_id: dict[int, dict[str, Any]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_id = int(raw.get("nodeID", len(nodes_by_id)))
        children = [int(value) for value in raw.get("children", []) if int(value) >= 0]
        node = dict(raw)
        node["nodeID"] = node_id
        node["children"] = children
        node["parent_id"] = int(raw["parent"]) if raw.get("parent", -1) not in (-1, None) else None
        node["depth"] = _stage_order(raw.get("nodetime"))
        node["genes"] = _node_genes(raw.get("genesInNode"), exgene)
        node["path_labels"] = []
        nodes_by_id[node_id] = node

    for node in nodes_by_id.values():
        parent_id = node.get("parent_id")
        node["parent"] = nodes_by_id.get(parent_id) if parent_id is not None else None

    leaves = sorted(
        [node for node in nodes_by_id.values() if not node.get("children")],
        key=lambda item: (-float(item.get("nodeMean", 0.0)), int(item["nodeID"])),
    )
    leaf_labels: dict[int, str] = {}
    for idx, leaf in enumerate(leaves):
        label = _letter_label(idx)
        leaf["path_labels"] = [label]
        leaf_labels[int(leaf["nodeID"])] = label

    def assign_paths(node: dict[str, Any]) -> list[str]:
        if node.get("path_labels"):
            return list(node["path_labels"])
        labels: list[str] = []
        for child_id in node.get("children", []):
            child = nodes_by_id.get(int(child_id))
            if child is None:
                continue
            labels.extend(assign_paths(child))
        deduped = sorted(set(labels))
        node["path_labels"] = deduped
        return deduped

    for node in nodes_by_id.values():
        assign_paths(node)
        node["path"] = ",".join(node["path_labels"])

    path_details: list[dict[str, Any]] = []
    path_curves: list[dict[str, Any]] = []
    for leaf in leaves:
        label = leaf_labels[int(leaf["nodeID"])]
        path_nodes: list[dict[str, Any]] = []
        current: dict[str, Any] | None = leaf
        while current is not None:
            path_nodes.append(current)
            current = current.get("parent")
        path_nodes.reverse()

        curve_points = [
            {
                "stage": str(node.get("nodetime", f"stage{node.get('depth', 0)}")),
                "stage_order": int(node.get("depth", 0)),
                "node_mean": float(node.get("nodeMean", 0.0)),
                "node_id": int(node["nodeID"]),
                "top_tfs": _node_top_tfs(node),
            }
            for node in path_nodes
        ]
        path_curves.append(
            {
                "label": f"Path {label}",
                "path_key": label,
                "stage_labels": [point["stage"] for point in curve_points],
                "stage_orders": [point["stage_order"] for point in curve_points],
                "values": [point["node_mean"] for point in curve_points],
                "point_details": curve_points,
                "hover_texts": [
                    "<br>".join(
                        [
                            f"Path: Path {label}",
                            f"Stage: {point['stage']}",
                            f"Mean: {point['node_mean']:.4f}",
                            (
                                "Top TFs: "
                                + ", ".join(
                                    (
                                        f"{item['name']} ({item['score']:.4g})"
                                        if item.get("score") is not None
                                        else item["name"]
                                    )
                                    for item in point["top_tfs"]
                                )
                            )
                            if point["top_tfs"]
                            else "Top TFs: none",
                        ]
                    )
                    for point in curve_points
                ],
            }
        )

        gene_rows = _path_gene_rows(
            leaf,
            gene_expression_stages=gene_expression_stages,
            gene_expression_lookup=gene_expression_lookup,
        )
        regulator_rows = _path_regulators(path_nodes)
        try:
            enrichment_table = enrichment_client.fetch([str(gene) for gene in leaf.get("genes", [])])
        except Exception as exc:
            enrichment_table = _empty_enrichment_table(f"ToppGene enrichment failed: {exc}")
        path_details.append(
            {
                "path_key": label,
                "label": f"Path {label}",
                "genes": _to_table(gene_rows),
                "regulators": _to_table(regulator_rows),
                "enrichment": enrichment_table,
            }
        )

    original_html = track_dir / "idrem_result.html"
    return {
        "track_key": track_key,
        "track_label": track_label or track_key,
        "track_dir": str(track_dir),
        "path_labels": [item["label"] for item in path_details],
        "selected_path": path_details[0]["path_key"] if path_details else None,
        "path_plot": {
            "stage_labels": path_curves[0]["stage_labels"] if path_curves else [],
            "paths": path_curves,
        },
        "paths": path_details,
        "original_html_path": str(original_html) if original_html.exists() else "",
    }


@dataclass
class _IdremSummaryRecord:
    fingerprint: tuple[Any, ...]
    summary: dict[str, Any]


class IdremTrackSummaryCache:
    def __init__(self, *, enrichment_client: ToppGeneEnrichmentClient | None = None) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], _IdremSummaryRecord] = {}
        self._enrichment_client = enrichment_client or ToppGeneEnrichmentClient()

    def get_summary(self, analysis_dir: str | Path, *, track_key: str, track_label: str = "") -> dict[str, Any]:
        analysis_path = Path(analysis_dir)
        track_dir = _resolve_idrem_track_dir(analysis_path, track_key)
        fingerprint = _track_fingerprint(analysis_path, track_dir)
        persisted_path = _summary_cache_path(analysis_path, track_key)
        persisted = _load_persisted_summary(persisted_path, fingerprint)
        if persisted is not None:
            if track_label:
                payload = dict(persisted)
                payload["track_label"] = track_label
                return payload
            return persisted
        fingerprint = (
            str(analysis_path.resolve()),
            str(track_dir.resolve()),
            _cache_key(track_dir / "DREM.json"),
            _cache_key(track_dir / "idrem_result.html"),
        )
        with self._lock:
            cached = self._cache.get(fingerprint)
            if cached is not None:
                return cached.summary
        summary = _build_summary(
            track_dir,
            track_key=track_key,
            track_label=track_label,
            enrichment_client=self._enrichment_client,
        )
        _persist_summary(persisted_path, _track_fingerprint(analysis_path, track_dir), summary)
        with self._lock:
            self._cache[fingerprint] = _IdremSummaryRecord(fingerprint=fingerprint, summary=summary)
        return summary


def precompute_analysis_idrem_summaries(
    analysis_dir: str | Path,
    *,
    overwrite: bool = False,
    enrichment_client: ToppGeneEnrichmentClient | None = None,
) -> dict[str, Any]:
    analysis_path = Path(analysis_dir).resolve()
    idrem_dir = analysis_path / "idrem"
    if not idrem_dir.exists():
        return {
            "analysis_dir": str(analysis_path),
            "summary_dir": str(_summary_cache_dir(analysis_path)),
            "track_count": 0,
            "generated": [],
            "skipped": [],
            "message": "No idrem directory was available for summary generation.",
        }
    client = enrichment_client or ToppGeneEnrichmentClient()
    generated: list[str] = []
    skipped: list[str] = []
    for track_dir in sorted(idrem_dir.glob("*.txt_viz")):
        if not track_dir.is_dir():
            continue
        track_key = track_dir.name[: -len(".txt_viz")]
        fingerprint = _track_fingerprint(analysis_path, track_dir)
        summary_path = _summary_cache_path(analysis_path, track_key)
        if not overwrite and _load_persisted_summary(summary_path, fingerprint) is not None:
            skipped.append(track_key)
            continue
        summary = _build_summary(
            track_dir,
            track_key=track_key,
            track_label=track_key,
            enrichment_client=client,
        )
        _persist_summary(summary_path, fingerprint, summary)
        generated.append(track_key)
    index_path = _summary_cache_dir(analysis_path) / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis_path),
                "track_count": len(generated) + len(skipped),
                "generated": generated,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "analysis_dir": str(analysis_path),
        "summary_dir": str(_summary_cache_dir(analysis_path)),
        "track_count": len(generated) + len(skipped),
        "generated": generated,
        "skipped": skipped,
        "index_path": str(index_path),
    }
