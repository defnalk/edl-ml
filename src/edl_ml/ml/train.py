"""Training loop for the capacitance surrogate with optional MLflow logging."""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from edl_ml.ml.dataset import LoaderBundle, StandardScalerTensor
from edl_ml.ml.model import CapacitanceMLP, MLPConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training hyperparameters.

    Parameters
    ----------
    learning_rate, weight_decay, max_epochs, batch_size
        Standard optimiser/loop controls.
    patience
        Early-stopping patience in epochs of unimproved validation loss.
    grad_clip
        Gradient-norm clip threshold; set to 0 to disable.
    device
        ``"auto"`` selects CUDA > MPS > CPU. Otherwise a torch device string.
    seed
        Seed for torch RNGs.
    mlflow_experiment
        If not ``None``, enable MLflow logging under this experiment.
    mlflow_tracking_uri
        Optional MLflow tracking URI; defaults to local ``mlruns/``.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    mlflow_experiment: str | None = None
    mlflow_tracking_uri: str | None = None


@dataclass(slots=True)
class TrainReport:
    """Summary of a training run.

    Attributes
    ----------
    model
        The trained model (loaded with the best validation checkpoint).
    best_val_loss
        Minimum validation loss observed.
    train_losses, val_losses
        Per-epoch training and validation mean squared error on the scaled
        target.
    test_metrics
        Dictionary of metrics evaluated on the test set in unscaled units
        (µF/cm²): ``mse``, ``rmse``, ``mae``, ``r2``, ``mape``.
    x_scaler, y_scaler
        Scalers used during training; persisted alongside the model for
        deployment-time inference.
    """

    model: CapacitanceMLP
    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    test_metrics: dict[str, float] = field(default_factory=dict)
    x_scaler: StandardScalerTensor | None = None
    y_scaler: StandardScalerTensor | None = None


def _resolve_device(name: str) -> torch.device:
    """Resolve the ``"auto"`` shorthand to a concrete device.

    ``auto`` prefers CUDA, then CPU. MPS is not selected automatically because
    the current PyTorch MPS backend still produces NaNs for small-batch
    training runs with ``AdamW`` under certain kernel launches; pass
    ``device="mps"`` explicitly to opt in.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_pass(
    model: CapacitanceMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> float:
    """Run one pass over ``loader``; returns the mean loss."""
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
        total += float(loss.item()) * x.shape[0]
        count += int(x.shape[0])
    return total / max(1, count)


def _evaluate_unscaled(
    model: CapacitanceMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    y_scaler: StandardScalerTensor,
    device: torch.device,
) -> dict[str, float]:
    """Compute regression metrics on unscaled capacitance (µF/cm²)."""
    model.eval()
    preds: list[NDArray[np.float32]] = []
    trues: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).cpu()
            p = y_scaler.inverse_transform(p)
            y_unscaled = y_scaler.inverse_transform(y)
            preds.append(p.numpy())
            trues.append(y_unscaled.numpy())
    p_arr = np.concatenate(preds, axis=0).ravel()
    t_arr = np.concatenate(trues, axis=0).ravel()
    err = p_arr - t_arr
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    mape = float(np.mean(np.abs(err / np.clip(np.abs(t_arr), 1e-8, None))) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_model(
    loaders: LoaderBundle,
    model_config: MLPConfig,
    train_config: TrainConfig,
    *,
    checkpoint_path: Path | str | None = None,
) -> TrainReport:
    """Train the capacitance MLP and return a :class:`TrainReport`.

    Supports optional MLflow logging. A best-on-validation checkpoint is kept
    in memory and optionally serialised to ``checkpoint_path``. Early
    stopping triggers when the validation loss fails to improve for
    ``train_config.patience`` consecutive epochs.
    """
    _set_seed(train_config.seed)
    device = _resolve_device(train_config.device)

    model = CapacitanceMLP(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, train_config.max_epochs))
    loss_fn = nn.MSELoss()

    mlflow_ctx = _maybe_mlflow(train_config, model_config)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = train_config.patience
    report = TrainReport(
        model=model,
        best_val_loss=float("inf"),
        x_scaler=loaders.x_scaler,
        y_scaler=loaders.y_scaler,
    )

    with mlflow_ctx as run:
        if run is not None:
            _log_params(run, model_config, train_config)

        for epoch in range(train_config.max_epochs):
            train_loss = _epoch_pass(
                model,
                loaders.train,
                loss_fn,
                device,
                optimizer=optimizer,
                grad_clip=train_config.grad_clip,
            )
            val_loss = _epoch_pass(model, loaders.val, loss_fn, device)
            scheduler.step()
            report.train_losses.append(train_loss)
            report.val_losses.append(val_loss)
            if run is not None:
                _log_metrics(
                    run,
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                )
            logger.info(
                "epoch=%d train=%.6f val=%.6f lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
            )
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = train_config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        report.best_val_loss = best_val
        report.test_metrics = _evaluate_unscaled(model, loaders.test, loaders.y_scaler, device)
        if run is not None:
            _log_metrics(run, {f"test_{k}": v for k, v in report.test_metrics.items()})

    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, model, loaders, model_config, train_config)
    return report


def _save_checkpoint(
    path: Path | str,
    model: CapacitanceMLP,
    loaders: LoaderBundle,
    model_config: MLPConfig,
    train_config: TrainConfig,
) -> None:
    """Persist model weights, scalers and configs to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_scaler": {
                "mean": loaders.x_scaler.mean,
                "std": loaders.x_scaler.std,
            },
            "y_scaler": {
                "mean": loaders.y_scaler.mean,
                "std": loaders.y_scaler.std,
            },
            "model_config": dataclasses.asdict(model_config),
            "train_config": dataclasses.asdict(train_config),
        },
        out,
    )


class _NullRun:
    """Context manager stand-in when MLflow is disabled."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _maybe_mlflow(train_config: TrainConfig, model_config: MLPConfig):  # type: ignore[no-untyped-def]
    if train_config.mlflow_experiment is None:
        return _NullRun()
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        logger.warning("mlflow requested but not installed; skipping logging")
        return _NullRun()
    if train_config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(train_config.mlflow_tracking_uri)
    mlflow.set_experiment(train_config.mlflow_experiment)
    return mlflow.start_run()


def _log_params(run: object, model_config: MLPConfig, train_config: TrainConfig) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    mlflow.log_params(
        {
            **{f"model.{k}": v for k, v in dataclasses.asdict(model_config).items()},
            **{f"train.{k}": v for k, v in dataclasses.asdict(train_config).items()},
        }
    )


def _log_metrics(run: object, metrics: dict[str, float], step: int | None = None) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)
