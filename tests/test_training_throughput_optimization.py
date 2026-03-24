import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData

import UNAGI.train.runner as runner_module
from UNAGI.train.runner import UNAGI_runner
from UNAGI.train.trainer import UNAGI_trainer
from UNAGI.dynamic_regulatory_networks.processIDREM import _default_idrem_worker_cap


class _FakeRunnerTrainer:
    def __init__(self):
        self.calls = 0
        self.modelName = "fake_model"

    def get_latent_representation(self, adata, iteration, target_dir):
        self.calls += 1
        n_obs = adata.n_obs
        embeddings = np.arange(n_obs * 2, dtype=np.float32).reshape(n_obs, 2)
        return embeddings.copy(), np.ones_like(embeddings), embeddings


class _TinyModel(torch.nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.linear = torch.nn.Linear(3, latent_dim)
        self.loaded = 0

    def load_state_dict(self, state_dict, strict=True):
        self.loaded += 1
        return super().load_state_dict(state_dict, strict=strict)

    def encoder(self, x, idx=None):
        if idx is not None:
            x = x[idx, :]
        mu = self.linear(x.float())
        logvar = torch.zeros_like(mu)
        return mu, logvar


def _make_stage_adata():
    adata = AnnData(
        np.array(
            [
                [1.0, 0.0, 0.5],
                [0.8, 0.2, 0.4],
                [0.1, 0.9, 0.3],
            ],
            dtype=np.float32,
        )
    )
    adata.obs = pd.DataFrame(
        {
            "label": ["A", "A", "B"],
        },
        index=["cell0", "cell1", "cell2"],
    )
    adata.var = pd.DataFrame(index=["G0", "G1", "G2"])
    adata.obsm["X_pca"] = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.2],
        ],
        dtype=np.float32,
    )
    adata.obsp["gcn_connectivities"] = sp.csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    return adata


def test_runner_writes_training_stage_timings(tmp_path):
    data_path = tmp_path / "data"
    (data_path / "0").mkdir(parents=True, exist_ok=True)
    trainer = _FakeRunnerTrainer()
    runner = UNAGI_runner(
        data_path,
        total_stage=2,
        iteration=0,
        trainer=trainer,
        label_key="label",
        idrem_dir=tmp_path / "idrem",
    )
    runner.load_stage_data = lambda: None
    runner.trainer.train = lambda *args, **kwargs: None
    runner.update_cell_attributes = lambda *args, **kwargs: None
    runner.build_temporal_dynamics_graph = lambda: None
    runner.run_IDREM = lambda: None
    runner.update_gene_weights_table = lambda: None
    runner.build_iteration_dataset = lambda: None
    runner.all_in_one = _make_stage_adata()

    runner.run(CPO=False)

    timings_path = data_path / "0" / "training_stage_timings.json"
    payload = json.loads(timings_path.read_text(encoding="utf-8"))
    assert payload["iteration"] == 0
    assert "training_model" in payload["stages"]
    assert "run_idrem" in payload["stages"]


def test_annotate_stage_data_reuses_cached_latent_and_disables_plots(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    (data_path / "0" / "stagedata").mkdir(parents=True, exist_ok=True)
    (data_path / "model_save").mkdir(parents=True, exist_ok=True)
    (data_path / "model_save" / "fake_model_0.pth").write_bytes(b"checkpoint")

    trainer = _FakeRunnerTrainer()
    runner = UNAGI_runner(
        data_path,
        total_stage=1,
        iteration=0,
        trainer=trainer,
        label_key="label",
        idrem_dir=tmp_path / "idrem",
    )
    adata = _make_stage_adata()
    runner.resolutions = [1.0]
    plot_calls = {"paga": 0, "umap": 0}

    monkeypatch.setattr(runner_module.sc.pp, "neighbors", lambda *args, **kwargs: None)

    def _fake_leiden(adata, resolution):
        adata.obs["leiden"] = pd.Categorical(["0"] * adata.n_obs)

    monkeypatch.setattr(runner_module.sc.tl, "leiden", _fake_leiden)
    monkeypatch.setattr(runner_module.sc.tl, "paga", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_module.sc.tl, "umap", lambda adata, **kwargs: adata.obsm.__setitem__("X_umap", np.zeros((adata.n_obs, 2))))
    monkeypatch.setattr(runner_module.sc.pl, "paga", lambda *args, **kwargs: plot_calls.__setitem__("paga", plot_calls["paga"] + 1))
    monkeypatch.setattr(runner_module.sc.pl, "umap", lambda *args, **kwargs: plot_calls.__setitem__("umap", plot_calls["umap"] + 1))
    monkeypatch.setattr(
        runner_module,
        "updateAttributes",
        lambda adata, rep: (adata, [np.zeros(adata.n_vars)], rep),
    )

    runner.neighbor_parameters = None
    runner.annotate_stage_data(adata, stage=0, CPO=True, max_neighbors=2)
    runner.neighbor_parameters = [2]
    runner.annotate_stage_data(adata, stage=0, CPO=True, max_neighbors=2)

    assert trainer.calls == 1
    assert plot_calls == {"paga": 0, "umap": 0}


def test_trainer_reuses_loaded_checkpoint(monkeypatch, tmp_path):
    target_dir = tmp_path / "workspace"
    (target_dir / "model_save").mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "model_save" / "model_0.pth"

    model = _TinyModel()
    torch.save(model.state_dict(), checkpoint_path)
    trainer = UNAGI_trainer(
        model=model,
        dis_model=_TinyModel(),
        modelName="model",
        batch_size=2,
        epoch_initial=1,
        epoch_iter=1,
        device=torch.device("cpu"),
        lr=1e-3,
        lr_dis=1e-3,
        GCN=False,
        cuda=False,
    )
    adata = _make_stage_adata()

    load_count = {"count": 0}
    original_torch_load = torch.load

    def _counting_load(*args, **kwargs):
        load_count["count"] += 1
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _counting_load)

    trainer.get_latent_representation(adata, 0, target_dir)
    trainer.get_latent_representation(adata, 0, target_dir)

    assert load_count["count"] == 1


def test_default_idrem_worker_cap_is_bounded(monkeypatch):
    monkeypatch.setattr("UNAGI.dynamic_regulatory_networks.processIDREM._available_memory_gb", lambda: 64)
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    assert _default_idrem_worker_cap(100) == 4
