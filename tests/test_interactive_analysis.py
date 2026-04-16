from __future__ import annotations

import matplotlib
import pickle
import pandas as pd
import UNAGI.monitoring.idrem_summary as idrem_summary_module
import UNAGI.monitoring.interactive_analysis as interactive_analysis_module

from UNAGI.monitoring.interactive_analysis import AnalysisExplorerService

from .interactive_analysis_test_helpers import write_analysis_dataset
from .monitor_test_helpers import fake_toppgene_enrichment_table, write_mock_idrem_track

matplotlib.use("Agg")


def test_explorer_metadata_discovers_hierarchical_dynamic_and_perturbation_options(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    view = explorer.build_view(analysis_dir)

    assert view["metadata"]["gene_count"] == 10
    assert view["active_mode"] == "dynamic_markers"
    assert view["umap"]["available"] is True
    assert view["metadata"]["umap_available"] is True
    assert view["metadata"]["umap_enabled"] is False
    assert all(option["value"] != "umap" for option in view["mode_options"])
    assert "stage" in view["umap"]["color_options"]
    assert view["hierarchical"]["available"] is True
    assert view["hierarchical"]["stage_options"][0]["value"] == "0"
    assert view["dynamic_markers"]["available"] is True
    assert view["dynamic_markers"]["ranked"]["track_options"][0]["value"] == "0-2-4"
    assert view["dynamic_markers"]["ranked"]["track_options"][0]["label"] == "track 0 AT1"
    assert view["dynamic_markers"]["genes"]["track_options"][0]["label"] == "track 0 AT1"
    assert view["perturbation"]["available"] is True
    assert "pathway" in view["perturbation"]["mode_options"]
    assert "two_drugs" in view["perturbation"]["mode_options"]
    assert view["perturbation"]["scope"] == "overall"
    assert view["perturbation"]["scope_options"] == ["overall"]
    assert view["perturbation"]["perturb_change_options"] == ["0.5"]
    assert view["perturbation"]["track_options"][0]["label"] == "Overall"


def test_explorer_discovers_and_renders_stage_specific_perturbation_results(tmp_path):
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter1",
        include_stage_specific_perturbation=True,
    )
    explorer = AnalysisExplorerService()

    view = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_scope": "stage",
            "pert_mode": "pathway",
            "pert_stage": "1",
            "pert_top_n": "2",
        },
    )

    assert view["perturbation"]["available"] is True
    assert view["perturbation"]["scope"] == "stage"
    assert view["perturbation"]["scope_options"] == ["overall", "stage"]
    assert view["perturbation"]["stage"] == "1"
    assert "two_drugs" in view["perturbation"]["mode_options"]
    assert [item["value"] for item in view["perturbation"]["stage_options"]] == ["1", "2"]
    assert view["perturbation"]["stage_options"][0]["label"] == "Stage 1"
    assert view["result"]["status"] == "ok"
    assert "scope=stage" in view["result"]["subtitle"]
    assert "Stage 1" in view["result"]["subtitle"]
    assert view["result"]["plots"][0]["publication_query"]["pert_scope"] == "stage"
    assert view["result"]["plots"][0]["table_download_query"]["pert_stage"] == "1"


def test_explorer_uses_grouped_overall_track_labels(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")

    explorer = AnalysisExplorerService()
    view = explorer.build_view(
        analysis_dir,
        form_data={"mode": "perturbation", "pert_mode": "pathway", "perturb_change": "0.5"},
    )

    labels = {
        item["value"]: item["label"]
        for item in view["perturbation"]["track_options"]
    }
    assert labels["0"] == "track 0 AT1"


def test_perturbation_track_options_do_not_repeat_overall(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    view = explorer.build_view(
        analysis_dir,
        form_data={"mode": "perturbation", "pert_mode": "pathway", "perturb_change": "0.5"},
    )

    overall_options = [
        item for item in view["perturbation"]["track_options"]
        if item["value"] == "overall"
    ]
    assert overall_options == [{"value": "overall", "label": "Overall"}]


def test_explorer_renders_stage_specific_track_perturbation_results(tmp_path, monkeypatch):
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter1",
        include_stage_specific_perturbation=True,
    )
    stage_dir = analysis_dir / "stage_specific_perturbation" / "stage_1"
    track_table_path = stage_dir / "pathway_perturbation_track_group_0_stage_1.csv"

    def fake_generate_stage_track_table(analysis_dir_arg, *, mode, stage, track_group):
        frame = pd.DataFrame(
            {
                "rank": [0, 1],
                "pathways": ["PathTrackA", "PathTrackB"],
                "perturbation score": [1.8, 1.1],
                "pval_adjusted": [0.003, 0.02],
                "regulated genes": ["['GENE0']", "['GENE1']"],
                "idrem_suggestion": ["['GENE0:+']", "['GENE1:-']"],
            }
        )
        track_table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(track_table_path, index=False)
        return str(track_table_path)

    monkeypatch.setattr(
        interactive_analysis_module,
        "_ensure_stage_specific_track_group_perturbation_table",
        fake_generate_stage_track_table,
    )

    explorer = AnalysisExplorerService()
    view = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_scope": "stage",
            "pert_mode": "pathway",
            "pert_stage": "1",
            "pert_track": "0",
            "pert_top_n": "2",
        },
    )

    assert view["perturbation"]["scope"] == "stage"
    assert view["perturbation"]["track"] == "0"
    assert view["perturbation"]["track_label"] == "track 0 AT1"
    assert view["perturbation"]["track_options"][0]["value"] == "overall"
    assert any(item["value"] == "0" for item in view["perturbation"]["track_options"])
    assert view["result"]["status"] == "ok"
    assert "track 0 AT1" in view["result"]["subtitle"]
    assert view["result"]["plots"][0]["publication_query"]["pert_track"] == "0"
    assert view["result"]["plots"][0]["table_download_query"]["pert_track"] == "0"


def test_stage_specific_track_dropdown_normalizes_group_cache_labels(tmp_path):
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter1",
        include_stage_specific_perturbation=True,
    )
    stage_dir = analysis_dir / "stage_specific_perturbation" / "stage_1"
    grouped_table_path = stage_dir / "pathway_perturbation_track_group_0_stage_1.csv"
    grouped_frame = pd.DataFrame(
        {
            "rank": [0, 1],
            "pathways": ["PathTrackA", "PathTrackB"],
            "perturbation score": [1.8, 1.1],
            "pval_adjusted": [0.003, 0.02],
        }
    )
    grouped_frame.to_csv(grouped_table_path, index=False)

    explorer = AnalysisExplorerService()
    view = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_scope": "stage",
            "pert_mode": "pathway",
            "pert_stage": "1",
        },
    )

    track_labels = {item["value"]: item["label"] for item in view["perturbation"]["track_options"]}
    assert "group_0" not in track_labels
    assert track_labels["0"] == "track 0 AT1"


def test_explorer_backfills_overall_track_perturbation_results_for_existing_analysis(tmp_path, monkeypatch):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    attribute_path = analysis_dir / "attribute.pkl"
    with attribute_path.open("rb") as handle:
        uns_payload = pickle.load(handle)
    uns_payload["drug_perturbation_score"]["0.5"] = {
        "overall": uns_payload["drug_perturbation_score"]["0.5"]["overall"]
    }
    with attribute_path.open("wb") as handle:
        pickle.dump(uns_payload, handle)

    track_table_path = analysis_dir / "overall_track_perturbation" / "drug_change_0.5" / "drug_perturbation_track_group_0.csv"

    def fake_generate_overall_track_tables(analysis_dir_arg, *, mode, perturb_change):
        frame = pd.DataFrame(
            {
                "compound": ["DrugTrackA", "DrugTrackB"],
                "perturbation score": [2.2, 1.4],
                "pval_adjusted": [0.004, 0.03],
            }
        )
        track_table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(track_table_path, index=False)
        return {"0": str(track_table_path)}

    monkeypatch.setattr(
        interactive_analysis_module,
        "_generate_overall_track_group_perturbation_tables",
        fake_generate_overall_track_tables,
    )

    explorer = AnalysisExplorerService()
    view = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_mode": "drug",
            "perturb_change": "0.5",
            "pert_track": "0",
            "pert_top_n": "2",
        },
    )

    assert any(item["value"] == "0" for item in view["perturbation"]["track_options"])
    assert view["perturbation"]["track"] == "0"
    assert view["result"]["status"] == "ok"
    assert "track 0 AT1" in view["result"]["subtitle"]


def test_explorer_renders_all_supported_modes(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    umap = explorer.build_view(analysis_dir, form_data={"mode": "umap", "enable_umap": "true"})
    assert umap["result"]["status"] == "ok"
    assert umap["umap"]["stage"] == "0"
    assert umap["umap"]["stage_options"] == ["0", "1", "2"]
    assert umap["result"]["umap_widget"]["default_color_by"] in umap["result"]["umap_widget"]["color_options"]
    assert umap["result"]["umap_widget"]["gene"] == ""
    assert umap["result"]["umap_widget"]["color_mode"] == "metadata"

    umap_gene = explorer.build_view(analysis_dir, form_data={"mode": "umap", "enable_umap": "true", "umap_gene": "gene0"})
    assert umap_gene["result"]["umap_widget"]["gene"] == "GENE0"
    assert umap_gene["result"]["umap_widget"]["color_mode"] == "gene"
    assert "color_by=gene (GENE0)" in umap_gene["result"]["subtitle"]

    hierarchical = explorer.build_view(
        analysis_dir,
        form_data={"mode": "hierarchical", "hier_stage": "0", "hier_cluster": "0", "hier_level": "0", "hier_top_n": "2"},
    )
    assert hierarchical["result"]["status"] == "ok"
    assert hierarchical["hierarchical"]["direction"] == "up"
    assert "direction=" not in hierarchical["result"]["subtitle"]
    assert hierarchical["result"]["plots"] == []
    assert hierarchical["result"]["split_cards"] == []
    assert not hierarchical["result"]["tables"]
    assert hierarchical["result"]["hierarchical_widget"]["table_download_query"]["mode"] == "hierarchical"

    dynamic = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "ranked_markers",
            "dyn_track": "0-2-4",
            "dyn_directions": ["increasing", "decreasing"],
            "dyn_plot_kind": "heatmap",
            "dyn_top_n": "3",
            "dyn_normalize_to_stage0": "true",
        },
    )
    assert dynamic["result"]["status"] == "ok"
    assert "track 0 AT1" in dynamic["result"]["subtitle"]
    assert dynamic["result"]["tables"] == []
    assert dynamic["result"]["plots"][0]["publication_query"]["dyn_track"] == "0-2-4"
    assert dynamic["result"]["plots"][0]["table_download_query"]["dyn_track"] == "0-2-4"

    custom = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "progression_track",
            "gene_track": "0-2-4",
            "gene_list": "gene0, geNe1",
            "gene_normalize_to_stage0": "true",
        },
    )
    assert custom["result"]["status"] == "ok"
    assert custom["dynamic_markers"]["genes"]["show_track"] is True
    assert custom["dynamic_markers"]["genes"]["scope_label"] == "temporal track"
    assert "track 0 AT1" in custom["result"]["subtitle"]
    assert "GENE0" in custom["result"]["subtitle"]
    assert "GENE1" in custom["result"]["subtitle"]
    assert "scope=temporal track" in custom["result"]["subtitle"]
    assert custom["result"]["plots"][0]["publication_query"]["gene_track"] == "0-2-4"
    assert custom["result"]["plots"][0]["table_download_query"]["gene_track"] == "0-2-4"
    assert custom["result"]["tables"] == []

    custom_global = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "global_stage",
            "gene_list": "GENE0, GENE1",
        },
    )
    assert custom_global["dynamic_markers"]["genes"]["show_track"] is False

    perturbation = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_mode": "drug",
            "perturb_change": "0.5",
            "pert_track": "overall",
            "pert_top_n": "2",
        },
    )
    assert perturbation["result"]["status"] == "ok"
    assert "Overall" in perturbation["result"]["subtitle"]
    assert perturbation["result"]["tables"] == []
    assert perturbation["result"]["plots"][0]["table_download_query"]["mode"] == "perturbation"
    assert 340 <= perturbation["result"]["plots"][0]["plotly_spec"]["layout"]["margin"]["l"] <= 460
    assert all(label.count("<br>") <= 1 for label in perturbation["result"]["plots"][0]["plotly_spec"]["data"][0]["y"])
    long_spec = interactive_analysis_module._perturbation_plotly_spec(
        pd.DataFrame(
            {
                "pathways": [
                    "HALLMARK_TGF_BETA_SIGNALING_VIA_SMAD_AND_NON_CANONICAL_BRANCHES",
                    "REACTOME_SIGNALING_BY_TGF_BETA_RECEPTOR_COMPLEX",
                ],
                "perturbation score": [0.8, 0.6],
                "pval_adjusted": [1e-4, 2e-3],
            }
        ),
        title="long labels",
    )
    y_labels = long_spec["data"][0]["y"]
    assert any("<br>" in str(label) for label in y_labels)
    assert all("…" not in str(label) for label in y_labels)
    assert 340 <= long_spec["layout"]["margin"]["l"] <= 460


def test_temporal_graph_payload_exposes_clickable_node_metadata(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    payload = explorer.get_temporal_graph_payload(analysis_dir)

    assert payload["node_count"] >= 2
    assert payload["edge_count"] >= 1
    assert payload["stage_tick_labels"] == ["Stage 0", "Stage 1", "Stage 2"]
    first_node = payload["nodes"][0]
    assert {"stage", "cluster", "cell_type", "n_cells", "x", "y"} <= set(first_node)

    stage_composition = explorer.get_stage_composition_payload(analysis_dir)
    assert stage_composition["data"][0]["type"] == "bar"
    assert stage_composition["layout"]["barmode"] == "stack"
    assert stage_composition["stage_order"] == ["0", "1", "2"]


def test_hierarchical_plotly_specs_and_idrem_track_summary_are_available(tmp_path, monkeypatch):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    write_mock_idrem_track(analysis_dir, track_key="0-2-4")
    monkeypatch.setattr(
        idrem_summary_module.ToppGeneEnrichmentClient,
        "fetch",
        lambda self, genes: fake_toppgene_enrichment_table(),
    )
    explorer = AnalysisExplorerService()
    temporal_payload = explorer.get_temporal_graph_payload(analysis_dir)
    expected_color_lookup = {
        (str(node["stage"]), str(node["cluster"])): str(node["color"])
        for node in temporal_payload["nodes"]
    }

    dendrogram = explorer.get_hierarchical_dendrogram_spec(analysis_dir, stage="0", cluster="0")
    assert dendrogram["data"][0]["type"] == "scatter"
    assert dendrogram["data"][1]["customdata"][0] == "0"
    assert dendrogram["data"][1]["marker"]["color"][0] == expected_color_lookup[("0", "0")]
    assert "Hierarchical dendrogram" in dendrogram["layout"]["title"]["text"]

    heatmap = explorer.get_hierarchical_heatmap_spec(
        analysis_dir,
        stage="0",
        cluster="0",
        level="0",
        direction="up",
        top_n=2,
    )
    assert heatmap["data"][0]["type"] == "heatmap"
    assert heatmap["data"][0]["y"][0] == "GENE0"

    summary = explorer.get_idrem_track_summary(analysis_dir, track_key="0-2-4", track_label="track 0 AT1")
    assert summary["track_label"] == "track 0 AT1"
    assert summary["path_labels"] == ["Path A", "Path B"]
    assert summary["paths"][0]["genes"]["columns"][0] == "Gene"
    assert summary["paths"][0]["regulators"]["columns"][0] == "Regulator"
    assert summary["paths"][0]["enrichment"]["columns"][0] == "Category"
    assert summary["paths"][0]["enrichment"]["row_count"] == 2
    assert summary["paths"][0]["enrichment"]["rows"][0][0] == "GO Biological Process"
    assert summary["paths"][0]["enrichment"]["rows"][0][3] == "2.0000e-03"
    assert summary["paths"][0]["enrichment"]["rows"][0][5] == "GENE0"
    assert summary["paths"][0]["enrichment"]["hover_texts"][0] == "Matched genes: GENE0"
    assert "Top TFs:" in summary["path_plot"]["paths"][0]["hover_texts"][1]


def test_idrem_summaries_can_be_precomputed_and_reused(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis" / "iter1"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_mock_idrem_track(analysis_dir, track_key="0-2-4")
    monkeypatch.setattr(
        idrem_summary_module.ToppGeneEnrichmentClient,
        "fetch",
        lambda self, genes: fake_toppgene_enrichment_table(),
    )
    result = idrem_summary_module.precompute_analysis_idrem_summaries(analysis_dir)
    assert result["track_count"] == 1
    summary_path = analysis_dir / "idrem_track_summaries" / "0-2-4.json"
    assert summary_path.exists()

    monkeypatch.setattr(
        idrem_summary_module.ToppGeneEnrichmentClient,
        "fetch",
        lambda self, genes: (_ for _ in ()).throw(AssertionError("persisted summary should be reused")),
    )
    cache = idrem_summary_module.IdremTrackSummaryCache()
    summary = cache.get_summary(analysis_dir, track_key="0-2-4", track_label="track 0 AT1")
    assert summary["track_label"] == "track 0 AT1"
    assert summary["paths"][0]["enrichment"]["rows"][0][5] == "GENE0"


def test_publication_pdf_export_supports_explorer_modes(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    for params in (
        {"mode": "umap", "figure_key": "umap", "umap_stage": "1", "umap_color_by": "ident"},
        {
            "mode": "hierarchical",
            "figure_key": "branch_heatmap",
            "hier_stage": "0",
            "hier_cluster": "0",
            "hier_level": "0",
            "hier_direction": "up",
            "hier_top_n": "2",
        },
        {
            "mode": "dynamic_markers",
            "figure_key": "heatmap",
            "dyn_view_mode": "ranked_markers",
            "dyn_track": "0-2-4",
            "dyn_directions": ["increasing"],
            "dyn_plot_kind": "heatmap",
        },
        {
            "mode": "dynamic_markers",
            "figure_key": "trajectory",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "progression_track",
            "gene_track": "0-2-4",
            "gene_list": "GENE0,GENE1",
        },
        {
            "mode": "perturbation",
            "figure_key": "ranked_bar",
            "pert_mode": "pathway",
            "perturb_change": "0.5",
            "pert_track": "overall",
        },
    ):
        pdf_bytes, filename = explorer.export_publication_pdf(analysis_dir, form_data=params)
        assert filename.endswith(".pdf")
        assert pdf_bytes.startswith(b"%PDF")

    csv_bytes, filename = explorer.export_table_csv(
        analysis_dir,
        form_data={
            "mode": "hierarchical",
            "hier_stage": "0",
            "hier_cluster": "0",
            "hier_level": "0",
            "hier_direction": "up",
            "hier_top_n": "2",
        },
    )
    assert filename.endswith(".csv")
    assert "AT1" in filename
    header = csv_bytes.decode("utf-8").splitlines()[0]
    assert header.split(",")[0] == "Gene"

    csv_bytes, filename = explorer.export_table_csv(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "ranked_markers",
            "dyn_track": "0-2-4",
            "dyn_directions": ["increasing", "decreasing"],
            "dyn_plot_kind": "heatmap",
            "dyn_top_n": "3",
            "dyn_normalize_to_stage0": "true",
        },
    )
    assert filename == "dynamic_markers_heatmap_track_0_AT1.csv"
    assert "stage0" in csv_bytes.decode("utf-8").splitlines()[0]

    csv_bytes, filename = explorer.export_table_csv(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "progression_track",
            "gene_track": "0-2-4",
            "gene_list": "GENE0,GENE1",
        },
    )
    assert filename == "custom_genes_progression_track_track_0_AT1.csv"
    assert "gene" in csv_bytes.decode("utf-8").splitlines()[0].lower()

    csv_bytes, filename = explorer.export_table_csv(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_mode": "pathway",
            "perturb_change": "0.5",
            "pert_track": "overall",
            "pert_top_n": "2",
        },
    )
    assert filename == "pathway_perturbation_overall_change_0.5.csv"
    assert "perturbation score" in csv_bytes.decode("utf-8").splitlines()[0]

    csv_bytes, filename = explorer.export_table_csv(
        analysis_dir,
        form_data={
            "mode": "perturbation",
            "pert_mode": "pathway",
            "perturb_change": "0.5",
            "pert_track": "0",
            "pert_top_n": "2",
        },
    )
    assert filename == "pathway_perturbation_track_0_AT1_change_0.5.csv"
    assert "perturbation score" in csv_bytes.decode("utf-8").splitlines()[0]

    stage_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter2",
        include_stage_specific_perturbation=True,
    )
    csv_bytes, filename = explorer.export_table_csv(
        stage_dir,
        form_data={
            "mode": "perturbation",
            "pert_scope": "stage",
            "pert_mode": "drug",
            "pert_stage": "2",
            "pert_top_n": "2",
        },
    )
    assert filename == "drug_perturbation_stage_2.csv"
    assert "DrugStage2A" in csv_bytes.decode("utf-8")

    grouped_stage_dir = stage_dir / "stage_specific_perturbation" / "stage_1"
    grouped_stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "rank": [0, 1],
            "pathways": ["PathTrackA", "PathTrackB"],
            "perturbation score": [1.8, 1.1],
            "pval_adjusted": [0.003, 0.02],
            "regulated genes": ["['GENE0']", "['GENE1']"],
            "idrem_suggestion": ["['GENE0:+']", "['GENE1:-']"],
        }
    ).to_csv(grouped_stage_dir / "pathway_perturbation_track_group_0_stage_1.csv", index=False)

    csv_bytes, filename = explorer.export_table_csv(
        stage_dir,
        form_data={
            "mode": "perturbation",
            "pert_scope": "stage",
            "pert_mode": "pathway",
            "pert_stage": "1",
            "pert_track": "0",
            "pert_top_n": "2",
        },
    )
    assert filename == "pathway_perturbation_stage_1_track_0_AT1.csv"
    assert "PathTrackA" in csv_bytes.decode("utf-8")


def test_custom_gene_validation_handles_missing_and_over_limit_requests(tmp_path):
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1")
    explorer = AnalysisExplorerService()

    mixed = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "global_stage",
            "gene_list": "gene0, MISSING1",
        },
    )
    assert mixed["result"]["status"] == "ok"
    assert mixed["result"]["warnings"]
    assert "MISSING1" in mixed["result"]["warnings"][0]

    too_many = explorer.build_view(
        analysis_dir,
        form_data={
            "mode": "dynamic_markers",
            "dyn_view_mode": "custom_genes",
            "gene_scope": "global_stage",
            "gene_list": ",".join(f"GENE{i}" for i in range(10)) + ",EXTRA1,EXTRA2,EXTRA3",
        },
    )
    assert too_many["result"]["status"] == "error"
    assert "at most 12 genes" in too_many["result"]["message"]


def test_unavailable_sections_return_clear_messages(tmp_path):
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter1",
        include_hierarchical=False,
        include_dynamic=False,
        include_perturbation=False,
        include_umap=False,
    )
    explorer = AnalysisExplorerService()

    umap = explorer.build_view(analysis_dir, form_data={"mode": "umap", "enable_umap": "true"})
    assert umap["result"]["status"] == "unavailable"
    assert "UMAP embedding" in umap["result"]["message"]

    hierarchical = explorer.build_view(analysis_dir, form_data={"mode": "hierarchical"})
    assert hierarchical["result"]["status"] == "unavailable"
    assert "ignore_hcmarkers" in hierarchical["result"]["message"]

    dynamic = explorer.build_view(analysis_dir, form_data={"mode": "dynamic_markers"})
    assert dynamic["result"]["status"] == "ok"
    assert "view=custom genes" in dynamic["result"]["subtitle"]

    dynamic_ranked = explorer.build_view(
        analysis_dir,
        form_data={"mode": "dynamic_markers", "dyn_view_mode": "ranked_markers"},
    )
    assert dynamic_ranked["result"]["status"] == "ok"
    assert "view=custom genes" in dynamic_ranked["result"]["subtitle"]

    perturbation = explorer.build_view(analysis_dir, form_data={"mode": "perturbation"})
    assert perturbation["result"]["status"] == "unavailable"
    assert "perturbation analysis was disabled" in perturbation["result"]["message"]


def test_umap_stage_options_adapt_to_arbitrary_stage_labels(tmp_path):
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter_custom",
        include_hierarchical=False,
        include_dynamic=False,
        include_perturbation=False,
        include_umap=True,
        stage_labels=("baseline", "day3", "day7", "day14", "day21"),
        cells_per_stage=5,
    )
    explorer = AnalysisExplorerService()

    view = explorer.build_view(analysis_dir, form_data={"mode": "umap", "umap_stage": "day14"})
    assert view["umap"]["stage_options"] == ["baseline", "day3", "day7", "day14", "day21"]
    assert view["umap"]["stage"] == "day14"

    payload = explorer.get_umap_payload(analysis_dir, stage="day21")
    assert payload["stage_options"] == ["baseline", "day3", "day7", "day14", "day21"]
    assert payload["stage"] == "day21"
    assert payload["point_count"] == 5
    assert all("cell_id" in point for point in payload["points"])
    assert all("x" in point and "y" in point for point in payload["points"])

    gene_payload = explorer.get_umap_payload(analysis_dir, stage="day21", gene="GENE0")
    assert gene_payload["expression_gene"] == "GENE0"
    assert "expression" in gene_payload["points"][0]
