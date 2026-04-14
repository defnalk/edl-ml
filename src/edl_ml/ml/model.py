"""Torch MLP surrogate for the Gouy-Chapman-Stern solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Hyperparameters of the capacitance MLP.

    Parameters
    ----------
    input_dim
        Number of input features. Equal to ``len(INPUT_COLUMNS)`` in the
        reference setup.
    hidden_dims
        Sequence of hidden-layer widths.
    activation
        One of ``"relu"``, ``"silu"`` or ``"gelu"``.
    dropout
        Probability for the dropout applied after each hidden layer.
    use_batch_norm
        Whether to insert a BatchNorm layer between linear and activation.
    """

    input_dim: int = 6
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    activation: str = "silu"
    dropout: float = 0.05
    use_batch_norm: bool = False

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if any(h < 1 for h in self.hidden_dims):
            raise ValueError("hidden_dims must be positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.activation not in {"relu", "silu", "gelu"}:
            raise ValueError(f"unknown activation: {self.activation}")


def _make_activation(name: str) -> nn.Module:
    """Return the activation module corresponding to ``name``."""
    return {
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
    }[name]


class CapacitanceMLP(nn.Module):
    """Fully connected regressor predicting the scaled capacitance.

    The network consumes a feature vector of physical parameters concatenated
    with the electrode potential and emits a single scalar (the standardised
    differential capacitance at that potential).

    Parameters
    ----------
    config
        Hyperparameters.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_make_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scaled capacitance prediction for input ``x``."""
        return self.net(x)

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
