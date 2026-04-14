"""Optuna hyperparameter search for the capacitance surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edl_ml.ml.dataset import LoaderBundle
from edl_ml.ml.model import MLPConfig
from edl_ml.ml.train import TrainConfig, train_model


@dataclass(frozen=True, slots=True)
class TuneConfig:
    """Controls for the Optuna study.

    Parameters
    ----------
    n_trials
        Number of hyperparameter trials.
    timeout_seconds
        Optional wall-clock cap. ``None`` means no cap.
    max_epochs
        Upper bound on training epochs inside each trial.
    patience
        Early-stopping patience inside each trial.
    study_name, storage
        Optuna study identifiers. ``storage=None`` uses an in-memory study.
    direction
        ``"minimize"`` (on val loss) or ``"maximize"`` (e.g. on -MSE).
    seed
        RNG seed controlling the TPE sampler.
    """

    n_trials: int = 40
    timeout_seconds: float | None = None
    max_epochs: int = 150
    patience: int = 15
    study_name: str = "edl-ml-surrogate"
    storage: str | None = None
    direction: str = "minimize"
    seed: int = 0


def _suggest_model_config(trial: Any, input_dim: int) -> MLPConfig:
    """Draw a random :class:`MLPConfig` from the Optuna trial."""
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    activation = trial.suggest_categorical("activation", ["relu", "silu", "gelu"])
    dropout = trial.suggest_float("dropout", 0.0, 0.25)
    use_bn = trial.suggest_categorical("batch_norm", [False, True])
    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=tuple(hidden for _ in range(n_layers)),
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_bn,
    )


def _suggest_train_config(trial: Any, base: TrainConfig) -> TrainConfig:
    """Derive a :class:`TrainConfig` from the Optuna trial."""
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    return TrainConfig(
        learning_rate=lr,
        weight_decay=wd,
        max_epochs=base.max_epochs,
        batch_size=base.batch_size,
        patience=base.patience,
        grad_clip=base.grad_clip,
        device=base.device,
        seed=base.seed,
        mlflow_experiment=base.mlflow_experiment,
        mlflow_tracking_uri=base.mlflow_tracking_uri,
    )


def run_optuna_study(
    loaders: LoaderBundle,
    input_dim: int,
    tune_config: TuneConfig,
    base_train_config: TrainConfig | None = None,
) -> Any:
    """Launch an Optuna TPE study optimising validation MSE.

    Parameters
    ----------
    loaders
        Pre-built data loaders (train/val/test + scalers).
    input_dim
        Number of input features.
    tune_config
        Study configuration.
    base_train_config
        Template training config; only learning-rate / weight-decay are
        overridden by the sampler.

    Returns
    -------
    optuna.Study
        The completed study; best trial accessible via ``.best_trial``.
    """
    import optuna

    base = base_train_config or TrainConfig(
        max_epochs=tune_config.max_epochs,
        patience=tune_config.patience,
    )

    def objective(trial: optuna.Trial) -> float:
        mcfg = _suggest_model_config(trial, input_dim)
        tcfg = _suggest_train_config(trial, base)
        report = train_model(loaders, mcfg, tcfg)
        return float(report.best_val_loss)

    sampler = optuna.samplers.TPESampler(seed=tune_config.seed)
    study = optuna.create_study(
        study_name=tune_config.study_name,
        storage=tune_config.storage,
        direction=tune_config.direction,
        load_if_exists=tune_config.storage is not None,
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=tune_config.n_trials,
        timeout=tune_config.timeout_seconds,
        gc_after_trial=True,
    )
    return study
