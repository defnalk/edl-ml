"""Generate the demo figures shown in the README.

Runs a small end-to-end pipeline and saves diagnostic plots to
``docs/assets``. Intended to be deterministic and quick (<1 minute).

Usage:
    python scripts/generate_demo_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from edl_ml.data import SamplingBounds, build_capacitance_dataset, split_by_sample  # noqa: E402
from edl_ml.ml import MLPConfig, TrainConfig, build_loaders, train_model  # noqa: E402
from edl_ml.ml.dataset import INPUT_COLUMNS  # noqa: E402
from edl_ml.ml.explain import permutation_feature_importance  # noqa: E402
from edl_ml.physics import GCSParameters, PBParameters, gouy_chapman_stern, solve_poisson_boltzmann  # noqa: E402
from edl_ml.viz import (  # noqa: E402
    plot_capacitance_curve,
    plot_error_distribution,
    plot_ion_profiles,
    plot_loss_curves,
    plot_parity,
)

OUT = Path("docs/assets")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    pb = solve_poisson_boltzmann(PBParameters(0.01, psi_diffuse_v=0.10))
    plot_ion_profiles(pb, title="Gouy-Chapman profile at c=10 mM, ψ_d=100 mV").savefig(
        OUT / "ion_profiles.png", dpi=180
    )

    params = GCSParameters(concentration_mol_l=0.1)
    E = np.linspace(-0.4, 0.4, 81)
    _, _, cap = gouy_chapman_stern(params, E)
    plot_capacitance_curve(
        E, cap * 100.0, title="GCS differential capacitance at 0.1 M"
    ).savefig(OUT / "cap_curve.png", dpi=180)

    df = build_capacitance_dataset(
        SamplingBounds(potential_n_points=81), n_samples=800, seed=0
    )
    train, val, test = split_by_sample(df)
    loaders = build_loaders(train, val, test, batch_size=512)
    report = train_model(
        loaders,
        MLPConfig(input_dim=len(INPUT_COLUMNS), hidden_dims=(256, 256, 128, 64), dropout=0.02),
        TrainConfig(max_epochs=300, learning_rate=2e-3, patience=40),
    )
    print("test metrics:", report.test_metrics)

    plot_loss_curves(report.train_losses, report.val_losses, title="Training curves").savefig(
        OUT / "loss_curves.png", dpi=180
    )

    import torch

    report.model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loaders.test:
            p = loaders.y_scaler.inverse_transform(report.model(x).cpu()).numpy()
            t = loaders.y_scaler.inverse_transform(y).numpy()
            preds.append(p)
            trues.append(t)
    y_pred = np.concatenate(preds).ravel()
    y_true = np.concatenate(trues).ravel()
    plot_parity(y_true, y_pred, title="Test parity (µF/cm²)").savefig(
        OUT / "parity.png", dpi=180
    )
    plot_error_distribution(y_true, y_pred, title="Test residuals (µF/cm²)").savefig(
        OUT / "residuals.png", dpi=180
    )

    feats = test[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    targets = test["capacitance_uf_cm2"].to_numpy(dtype=np.float32)
    importances = permutation_feature_importance(
        report.model, feats, targets, loaders.x_scaler, loaders.y_scaler, n_repeats=10
    )
    with (OUT / "permutation_importance.txt").open("w") as fh:
        fh.write("feature\timportance(ΔMAE µF/cm²)\n")
        for name, val in zip(INPUT_COLUMNS, importances, strict=True):
            fh.write(f"{name}\t{val:.4f}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
