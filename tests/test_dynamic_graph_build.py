from __future__ import annotations

from inspect import signature

from UNAGI.dynamic_graphs.buildGraph import connectNodes, _format_no_edges_error


def test_connect_nodes_uses_configurable_cutoff_with_default_005():
    distances = [[0.0, 0.05], [0.05, 0.1]]

    assert signature(connectNodes).parameters["cutoff"].default == 0.05
    assert connectNodes(distances, cutoff=0.05) == []
    assert connectNodes(distances, cutoff=0.1) == [[0, 0]]


def test_no_edges_error_reports_cutoff_filtered_candidates():
    distances = [[0.0, 0.05], [0.05, 0.1]]

    message = _format_no_edges_error(3, 4, distances, 0.05)

    assert "No dynamic graph edges passed connect_edges_cutoff=0.05 between stage 3 and stage 4." in message
    assert "All nearest-cluster candidate p-values were above the cutoff" in message
    assert "Best candidate was stage 3 cluster 0 -> stage 4 cluster 0" in message
    assert "Increase run.connect_edges_cutoff" in message
