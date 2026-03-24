from types import SimpleNamespace

import pandas as pd
import pytest

import UNAGI.UNAGI_tool as tool_module


class _FakeAnalyst:
    last_call = None

    def __init__(self, *args, **kwargs):
        self.adata = SimpleNamespace(
            uns={
                "single_gene_perturbation_score": {
                    "0.25": {
                        "overall": {
                            "top_compounds": {
                                "compound": {"0": "GENE1"},
                                "pval_adjusted": {"0": 0.01},
                            }
                        }
                    }
                }
            }
        )

    def perturbation_analyse_customized_pathway(self, *args, **kwargs):
        self.__class__.last_call = ("pathway", args, kwargs)

    def perturbation_analyse_customized_drug(self, *args, **kwargs):
        self.__class__.last_call = ("drug", args, kwargs)

    def perturbation_analyse_single_gene(self, *args, **kwargs):
        self.__class__.last_call = ("single_gene", args, kwargs)

    def perturbation_analyse_gene_combinatorial(self, *args, **kwargs):
        self.__class__.last_call = ("gene_combo", args, kwargs)

    def start_analyse(self, *args, **kwargs):
        self.__class__.last_call = ("analyse", args, kwargs)


@pytest.mark.parametrize(
    ("method_name", "args", "expected_tag"),
    [
        (
            "customize_pathway_perturbation",
            ("data.h5ad", 1, "pathway.npy"),
            "pathway",
        ),
        (
            "customize_drug_perturbation",
            ("data.h5ad", 1, "drug.npy"),
            "drug",
        ),
        (
            "single_gene_perturbation",
            ("data.h5ad", 1),
            "single_gene",
        ),
        (
            "gene_combinatorial_perturbation",
            ("data.h5ad", 1),
            "gene_combo",
        ),
    ],
)
def test_public_perturbation_apis_forward_bound_and_direction_mode(monkeypatch, method_name, args, expected_tag):
    monkeypatch.setattr(tool_module, "analyst", _FakeAnalyst)
    unagi = tool_module.UNAGI(random_seed=0)

    method = getattr(unagi, method_name)
    method(*args, perturb_change=0.25, direction_mode="one_sided")

    tag, _call_args, call_kwargs = _FakeAnalyst.last_call
    assert tag == expected_tag
    assert call_kwargs["bound"] == pytest.approx(0.25)
    assert call_kwargs["direction_mode"] == "one_sided"


def test_analyse_unagi_forwards_direction_mode(monkeypatch):
    monkeypatch.setattr(tool_module, "analyst", _FakeAnalyst)
    unagi = tool_module.UNAGI(random_seed=0)

    unagi.analyse_UNAGI(
        data_path="dataset.h5ad",
        iteration=1,
        random_background_sampling_times=10,
        run_perturbation=True,
        perturb_change=0.5,
        direction_mode="one_sided",
        perturbation_schedule="single_gpu_optimized",
    )

    tag, _call_args, call_kwargs = _FakeAnalyst.last_call
    assert tag == "analyse"
    assert call_kwargs["direction_mode"] == "one_sided"
    assert call_kwargs["perturbation_schedule"] == "single_gpu_optimized"
