import pandas as pd

from UNAGI.monitoring.explorer_cache import _build_palette as build_explorer_palette
from UNAGI.palettes import DEFAULT_CATEGORICAL_COLORS
from UNAGI.plotting import build_palette as build_plot_palette


def test_default_categorical_palette_exposes_50_unique_colors_before_repeating():
    assert len(DEFAULT_CATEGORICAL_COLORS) >= 50
    assert len(set(DEFAULT_CATEGORICAL_COLORS[:50])) == 50


def test_plotting_and_explorer_palettes_match_for_first_50_categories():
    categories = [str(index) for index in range(50)]

    plotting_palette = build_plot_palette(categories)
    explorer_palette = build_explorer_palette(pd.Index(categories))

    assert [plotting_palette[category] for category in categories] == list(DEFAULT_CATEGORICAL_COLORS[:50])
    assert plotting_palette == explorer_palette
