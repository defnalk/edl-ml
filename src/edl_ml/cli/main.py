"""``edl`` command line interface.

Implemented with :mod:`argparse` (stdlib only) so the CLI remains available
even when the optional rich-ecosystem dependencies are not installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from edl_ml._version import __version__

logger = logging.getLogger(__name__)


def _add_simulate(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("simulate", help="Run the Gouy-Chapman-Stern solver for one setup.")
    p.add_argument("--concentration", type=float, default=0.1, help="mol/L")
    p.add_argument("--valence", type=int, default=1)
    p.add_argument("--temperature", type=float, default=298.15, help="K")
    p.add_argument("--stern-thickness", type=float, default=3.0, help="angstrom")
    p.add_argument("--stern-permittivity", type=float, default=6.0)
    p.add_argument("--e-min", type=float, default=-0.4)
    p.add_argument("--e-max", type=float, default=0.4)
    p.add_argument("--n-points", type=int, default=81)
    p.add_argument("--output", type=Path, default=None)


def _add_generate(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("generate", help="Generate the training dataset.")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("data/processed/dataset.parquet"))
    p.add_argument("--no-parallel", action="store_true")


def _add_train(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("train", help="Train the MLP surrogate.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("data/models/model.pt"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)


def _add_tune(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("tune", help="Launch an Optuna hyperparameter search.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--out", type=Path, default=Path("data/models/best_config.json"))


def _add_evaluate(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("evaluate", help="Evaluate a checkpoint against the physics solver.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--figures-dir", type=Path, default=Path("data/reports/figures"))


def _add_predict(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("predict", help="Predict capacitance at a single operating point.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--concentration", type=float, required=True)
    p.add_argument("--valence", type=int, default=1)
    p.add_argument("--temperature", type=float, default=298.15)
    p.add_argument("--stern-thickness", type=float, default=3.0)
    p.add_argument("--stern-permittivity", type=float, default=6.0)
    p.add_argument("--e-min", type=float, default=-0.4)
    p.add_argument("--e-max", type=float, default=0.4)
    p.add_argument("--n-points", type=int, default=81)


def app() -> argparse.ArgumentParser:
    """Return the fully configured top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="edl",
        description="Electric double layer simulator with ML surrogate.",
    )
    parser.add_argument("--version", action="version", version=f"edl-ml {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_simulate(sub)
    _add_generate(sub)
    _add_train(sub)
    _add_tune(sub)
    _add_evaluate(sub)
    _add_predict(sub)
    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    from edl_ml.physics.gcs import GCSParameters, gouy_chapman_stern

    params = GCSParameters(
        concentration_mol_l=args.concentration,
        valence=args.valence,
        temperature_k=args.temperature,
        stern_thickness_m=args.stern_thickness * 1e-10,
        stern_permittivity=args.stern_permittivity,
    )
    potentials = np.linspace(args.e_min, args.e_max, args.n_points)
    sigma, psid, cap = gouy_chapman_stern(params, potentials)
    rows = [
        {
            "electrode_potential_v": float(e),
            "psi_diffuse_v": float(p),
            "surface_charge_uc_cm2": float(s) * 100.0,
            "capacitance_uf_cm2": float(c) * 100.0,
        }
        for e, p, s, c in zip(potentials, psid, sigma, cap, strict=True)
    ]
    payload = {"parameters": dataclasses.asdict(params), "data": rows}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", args.output)
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from edl_ml.data import SamplingBounds, build_capacitance_dataset
    from edl_ml.data.generate import save_dataset, summarise_dataset

    df = build_capacitance_dataset(
        SamplingBounds(),
        n_samples=args.n_samples,
        seed=args.seed,
        parallel=not args.no_parallel,
    )
    save_dataset(df, args.output)
    stats = summarise_dataset(df)
    logger.info("dataset stats: %s", stats)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from edl_ml.data import load_dataset, split_by_sample
    from edl_ml.ml import MLPConfig, TrainConfig, build_loaders, train_model

    df = load_dataset(args.data)
    train_df, val_df, test_df = split_by_sample(df, seed=args.seed)
    loaders = build_loaders(train_df, val_df, test_df, batch_size=args.batch_size)
    report = train_model(
        loaders,
        MLPConfig(input_dim=7),  # 6 features + electrode potential
        TrainConfig(
            learning_rate=args.lr,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            mlflow_experiment=args.mlflow_experiment,
            seed=args.seed,
        ),
        checkpoint_path=args.checkpoint,
    )
    logger.info("best val loss: %.6f", report.best_val_loss)
    logger.info("test metrics: %s", report.test_metrics)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    from edl_ml.data import load_dataset, split_by_sample
    from edl_ml.ml import TuneConfig, build_loaders, run_optuna_study

    df = load_dataset(args.data)
    train_df, val_df, test_df = split_by_sample(df)
    loaders = build_loaders(train_df, val_df, test_df)
    study = run_optuna_study(
        loaders,
        input_dim=7,
        tune_config=TuneConfig(
            n_trials=args.n_trials,
            timeout_seconds=args.timeout,
            storage=args.storage,
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study.best_trial.params, indent=2))
    logger.info("best value: %.6f", study.best_value)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import torch

    from edl_ml.data import load_dataset, split_by_sample
    from edl_ml.ml import CapacitanceMLP, MLPConfig, build_loaders
    from edl_ml.ml.dataset import INPUT_COLUMNS
    from edl_ml.ml.train import _evaluate_unscaled, _resolve_device
    from edl_ml.viz import plot_error_distribution, plot_parity

    df = load_dataset(args.data)
    train_df, val_df, test_df = split_by_sample(df)
    loaders = build_loaders(train_df, val_df, test_df)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MLPConfig(**ckpt["model_config"])
    model = CapacitanceMLP(cfg)
    model.load_state_dict(ckpt["state_dict"])
    device = _resolve_device("auto")
    model.to(device)
    metrics = _evaluate_unscaled(model, loaders.test, loaders.y_scaler, device)
    logger.info("test metrics: %s", metrics)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Parity + error dist on test split.
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loaders.test:
            p = loaders.y_scaler.inverse_transform(model(x.to(device)).cpu()).numpy()
            t = loaders.y_scaler.inverse_transform(y).numpy()
            preds.append(p)
            trues.append(t)
    y_pred = np.concatenate(preds).ravel()
    y_true = np.concatenate(trues).ravel()
    fig_p = plot_parity(y_true, y_pred, title="Test parity (µF/cm²)")
    fig_e = plot_error_distribution(y_true, y_pred, title="Test residuals (µF/cm²)")
    fig_p.savefig(args.figures_dir / "parity_test.png", dpi=200)
    fig_e.savefig(args.figures_dir / "residuals_test.png", dpi=200)
    logger.info("figures written to %s", args.figures_dir)
    _ = INPUT_COLUMNS  # silence unused-import warning when no SHAP is run
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from edl_ml.ml import CapacitanceMLP, MLPConfig
    from edl_ml.ml.dataset import StandardScalerTensor

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MLPConfig(**ckpt["model_config"])
    model = CapacitanceMLP(cfg)
    model.load_state_dict(ckpt["state_dict"])
    x_scaler = StandardScalerTensor(ckpt["x_scaler"]["mean"], ckpt["x_scaler"]["std"])
    y_scaler = StandardScalerTensor(ckpt["y_scaler"]["mean"], ckpt["y_scaler"]["std"])
    potentials = np.linspace(args.e_min, args.e_max, args.n_points)
    rows = np.column_stack(
        [
            np.full_like(potentials, np.log10(args.concentration)),
            np.full_like(potentials, float(args.valence)),
            np.full_like(potentials, float(args.temperature)),
            np.full_like(potentials, float(args.stern_thickness)),
            np.full_like(potentials, float(args.stern_permittivity)),
            potentials,
        ]
    )
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(rows, dtype=torch.float32)
        xt = x_scaler.transform(xt)
        out = model(xt)
        out = y_scaler.inverse_transform(out)
    cap = out.cpu().numpy().ravel()
    for e, c in zip(potentials, cap, strict=True):
        print(f"{float(e):+0.4f}\t{float(c):0.4f}")
    return 0


_COMMANDS = {
    "simulate": _cmd_simulate,
    "generate": _cmd_generate,
    "train": _cmd_train,
    "tune": _cmd_tune,
    "evaluate": _cmd_evaluate,
    "predict": _cmd_predict,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = app()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
