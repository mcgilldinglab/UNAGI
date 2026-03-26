from __future__ import annotations

import json
from pathlib import Path

from UNAGI.monitoring.share_bundle import export_analysis_share_bundle

from .interactive_analysis_test_helpers import write_analysis_dataset
from .monitor_test_helpers import fake_toppgene_enrichment_table, write_mock_idrem_track


def test_export_analysis_share_bundle_writes_index_and_zip(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis" / "iter1" / "results"
    write_analysis_dataset(analysis_dir, include_stage_specific_perturbation=True)
    write_mock_idrem_track(analysis_dir, track_key="0-2-4")
    monkeypatch.setattr(
        "UNAGI.monitoring.idrem_summary.ToppGeneEnrichmentClient.fetch",
        lambda self, genes: fake_toppgene_enrichment_table(),
    )

    manifest = export_analysis_share_bundle(analysis_dir, analysis_label="Mock Analysis")

    bundle_dir = analysis_dir / "share_bundle"
    index_path = bundle_dir / "index.html"
    stage_umap_path = bundle_dir / "assets" / "figures" / "stage_umap_grid_300dpi.png"
    stage_composition_pdf = bundle_dir / "assets" / "publication" / "stage_composition_publication.pdf"
    saved_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    zip_path = Path(saved_manifest["zip_path"])

    assert index_path.exists()
    assert zip_path.exists()
    assert stage_umap_path.exists()
    assert stage_composition_pdf.exists()
    assert manifest["analysis_label"] == "Mock Analysis"
    assert manifest["counts"]["entries"] > 0
    assert manifest["counts"]["tracks"] >= 1
    assert saved_manifest["zip_path"] == str(zip_path)
    assert zip_path.name.startswith("unagi_offline_report_Mock_Analysis_")

    html_text = index_path.read_text(encoding="utf-8")
    assert "Offline Share Bundle" in html_text
    assert "Temporal Dynamics Graph" in html_text
    assert "0-2-4" in html_text
    assert 'id="perturb-scope-pathway"' in html_text
    assert 'id="perturb-change-pathway"' in html_text
    assert 'id="perturb-stage-pathway"' in html_text
    assert 'id="perturb-track-pathway"' in html_text
    assert 'id="perturb-plot-pathway"' in html_text
    assert "track 0 AT1" in html_text
    assert "Individual stage" in html_text
    assert "Stage 1" in html_text
    assert "Download detailed table" in html_text
    assert "Download plot (PDF)" in html_text
    assert "stage_composition_publication.pdf" in html_text
    assert "Download enrichment details" in html_text
    assert 'id="dynamic-track"' in html_text
    assert 'id="hier-stage"' in html_text
    assert 'class="stack-grid"' in html_text
    assert "compactDynamicTable" in html_text
    assert "signed log fold changes for the selected branch against its sibling comparison clusters" in html_text
    assert "Log fold change" in html_text
    assert "Marker genes" in html_text
    assert "log fold changes" in html_text
    assert "formatAxisTick" in html_text
    assert 'class="track-section"' in html_text
    assert "<summary>Enrichment</summary>" in html_text
    assert "<summary>Regulators</summary>" in html_text
    assert "Biological process" in html_text
    assert 'data-enrichment-tab="' in html_text
    assert "Rank" in html_text
    assert "ranked by adjusted p-value" in html_text
    assert "report-table enrichment-table" in html_text
    assert "function formatEnrichmentTerm" in html_text
    assert "height - margin.bottom, margin.top" in html_text
